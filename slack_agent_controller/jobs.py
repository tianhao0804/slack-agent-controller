"""Tracks ``sac-job`` jobs and reports them in Slack.

Jobs are written by ``deploy/sac_job.py`` (the CLI and the runner inside each
job's systemd unit) under ``JOBS_DIR``. This watcher only reads those files:

* a new job gets one status message in the Slack thread of the Claude session
  that submitted it (terminal jobs: a new thread in the mirror channel, or a DM),
  with Log / Cancel buttons; the message is updated as the job moves on and
  every ``progress_min`` minutes while it runs (runtime, GPU use, last lines);
* lines that look like trouble (Traceback, nan, out of memory, ...) trigger
  one @mention per kind of problem;
* when the job ends, a new message @mentions you with the exit status, the
  end of the log, and buttons to attach the full log or let Claude look at it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from . import policy
from .slackfmt import button, escape, inline_code
from .store import _atomic_write, load_json_with_backup

if TYPE_CHECKING:
    from .app import SlackFrontend

log = logging.getLogger(__name__)

CLI = Path(__file__).resolve().parent.parent / "deploy" / "sac_job.py"
TAIL_LINES = 30
PROGRESS_LINES = 5
MAX_LOG_READ = 2_000_000
ALERTS = [
    ("traceback", re.compile(r"^Traceback \(most recent call last\)", re.M)),
    ("nan", re.compile(r"(?<![\w.])(?:nan|NaN)(?![\w.])")),
    ("oom", re.compile(r"out of memory|OutOfMemoryError|CUDA error|cudaErrorMemoryAllocation", re.I)),
    ("killed", re.compile(r"^Killed$|Segmentation fault|core dumped|Bus error", re.M)),
]
_ICON = {"queued": ":hourglass_flowing_sand:", "running": ":gear:", "finished": ":white_check_mark:",
         "failed": ":x:", "cancelled": ":stop_button:", "lost": ":warning:"}
_DONE = ("finished", "failed", "cancelled", "lost")


def load_cli(jobs_dir: Path) -> ModuleType:
    """deploy/sac_job.py is standalone (stdlib only, run with -I); import it by path."""
    spec = importlib.util.spec_from_file_location("sac_job", CLI)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sac_job", mod)
    spec.loader.exec_module(mod)
    mod.JOBS_DIR = jobs_dir  # type: ignore[attr-defined]
    return mod


def _dur(s: float) -> str:
    s = int(max(0, s))
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


class JobWatcher:
    def __init__(self, front: SlackFrontend) -> None:
        self.front = front
        self.hub = front.hub
        self.cfg = front.cfg
        self.dir = self.cfg.jobs_dir
        self.cli = load_cli(self.dir)
        self.path = self.cfg.state_dir / "jobs_watch.json"
        raw = load_json_with_backup(self.path)
        self.state: dict[str, dict[str, Any]] = raw if isinstance(raw, dict) else {}
        self._first_run = not self.path.exists()

    # -- setup ---------------------------------------------------------------
    def write_config(self) -> None:
        """Tell the CLI which directories jobs may run in."""
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.dir.chmod(0o700)
        _atomic_write(self.dir / "config.json", json.dumps({"roots": [str(r) for r in self.cfg.cwd_roots]}))

    def save(self) -> None:
        _atomic_write(self.path, json.dumps(self.state, indent=1), backup=True)

    async def loop(self) -> None:
        await asyncio.to_thread(self.write_config)
        while True:
            try:
                await self.scan()
            except Exception:
                log.exception("job scan failed")
            await asyncio.sleep(self.cfg.jobs_poll_s)

    # -- reading job files -----------------------------------------------------
    def _snapshot(self) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        self.cli.reap_lost()
        return [(j, self.cli.read_meta(j), self.cli.read_status(j)) for j in self.cli.all_jobs()]

    def _log_path(self, jid: str) -> Path:
        return self.dir / jid / "log"

    def tail(self, jid: str, n: int) -> str:
        f = self._log_path(jid)
        try:
            size = f.stat().st_size
            with f.open("rb") as fh:
                fh.seek(max(0, size - 64_000))
                lines = fh.read().decode(errors="replace").splitlines()
        except OSError:
            return ""
        lines = [ln for ln in lines if not ln.startswith("[sac-job ")] or lines
        return policy.redact_log("\n".join(lines[-n:]))

    def _new_log(self, jid: str, st: dict[str, Any]) -> str:
        f = self._log_path(jid)
        off = int(st.get("log_offset") or 0)
        try:
            size = f.stat().st_size
            if size < off:
                off = 0
            with f.open("rb") as fh:
                fh.seek(off)
                data = fh.read(MAX_LOG_READ)
        except OSError:
            return ""
        st["log_offset"] = off + len(data)
        return data.decode(errors="replace")

    # -- where to report -----------------------------------------------------------
    async def _destination(self, jid: str, meta: dict[str, Any]) -> tuple[str, str, str] | None:
        """(channel, thread_ts, owner): the submitting session's thread, else a new thread."""
        sid = meta.get("session_id")
        rec = self.hub.store.find_by_session(sid) if sid else None
        if rec is not None:
            return rec.channel, rec.thread_ts, rec.owner
        owner = self.cfg.mirror_user or sorted(self.cfg.allowed_users)[0]
        channel = self.cfg.mirror_channel or await self.front._dm_channel(owner)
        if not channel:
            return None
        where = "a terminal" if meta.get("origin") == "terminal" else "a session the bot does not know"
        root = await self.hub.post(channel, None, f":test_tube: *Job {escape(str(meta.get('name')))}* (`{jid}`) "
                                   f"submitted from {where} in {inline_code(str(meta.get('cwd')))}")
        return (channel, root, owner) if root else None

    # -- messages ------------------------------------------------------------------
    def _status_text(self, jid: str, meta: dict[str, Any], status: dict[str, Any], gpu_use: str = "") -> str:
        state = str(status.get("state", "?"))
        start = status.get("started")
        end = status.get("ended") or time.time()
        bits = [f"{_ICON.get(state, '')} *Job {escape(str(meta.get('name')))}* (`{jid}`) — *{state}*"]
        if status.get("gpus"):
            bits.append(f"GPU {','.join(map(str, status['gpus']))}{gpu_use}")
        elif meta.get("gpus"):
            bits.append(f"wants {meta['gpus']} GPU(s)")
        else:
            bits.append("CPU")
        if start:
            bits.append(_dur(end - start))
        if state == "queued" and status.get("reason"):
            bits.append(escape(str(status["reason"])))
        head = " · ".join(bits)
        cmd = " ".join(meta.get("command") or [])
        text = f"{head}\n{inline_code(policy.redact_log(cmd)[:200])} in {inline_code(str(meta.get('cwd')))}"
        if state == "running":
            t = self.tail(jid, PROGRESS_LINES)
            if t.strip():
                text += f"\n```{escape(t[-1500:])}```"
            text += f"\n_updated {time.strftime('%H:%M')} · every {meta.get('progress_min') or 30} min_"
        return text[:2900]

    def _status_blocks(self, jid: str, text: str, state: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
        elems = [button(":page_facing_up: Log", "sac_job_log", jid)]
        if state in ("queued", "running"):
            cancel = button(":stop_button: Cancel job", "sac_job_cancel", jid, "danger")
            cancel["confirm"] = {"title": {"type": "plain_text", "text": "Cancel this job?"},
                                 "text": {"type": "plain_text", "text": f"Stop job {jid} and everything it started."},
                                 "confirm": {"type": "plain_text", "text": "Cancel job"},
                                 "deny": {"type": "plain_text", "text": "Keep running"}}
            elems.append(cancel)
        blocks.append({"type": "actions", "block_id": f"sac_job_{jid}", "elements": elems})
        return blocks

    async def _gpu_use(self, gpus: list[int]) -> str:
        rows = await asyncio.to_thread(self.cli.gpu_state)
        use = [f"{int(u)}%" for i, u, _m in rows if i in gpus]
        return f" ({'/'.join(use)} busy)" if use else ""

    async def _update(self, jid: str, st: dict[str, Any], meta: dict[str, Any], status: dict[str, Any]) -> None:
        gpu_use = await self._gpu_use(status.get("gpus") or []) if status.get("state") == "running" else ""
        text = self._status_text(jid, meta, status, gpu_use)
        blocks = self._status_blocks(jid, text, str(status.get("state")))
        try:
            if st.get("msg_ts"):
                await self.hub.web.chat_update(channel=st["channel"], ts=st["msg_ts"], text=text, blocks=blocks)
            else:
                st["msg_ts"] = await self.hub.post(st["channel"], st["thread_ts"], text, blocks=blocks)
        except Exception:
            log.exception("job %s: status update failed", jid)

    async def _finished(self, jid: str, st: dict[str, Any], meta: dict[str, Any], status: dict[str, Any]) -> None:
        state = str(status.get("state"))
        took = _dur((status.get("ended") or time.time()) - status["started"]) if status.get("started") else "-"
        what = {"finished": "finished", "failed": f"failed (exit {status.get('exit_code')})",
                "cancelled": "was cancelled", "lost": "was lost (its process is gone, e.g. after a node reboot)"}
        head = (f"<@{st['owner']}> {_ICON.get(state, '')} Job *{escape(str(meta.get('name')))}* (`{jid}`) "
                f"{what.get(state, state)} after {took}.")
        t = self.tail(jid, TAIL_LINES)
        body = head + (f"\n```{escape(t[-2600:])}```" if t.strip() else "\n_(no output)_")
        elems = [button(":paperclip: Full log", "sac_job_full", jid)]
        if self.hub.store.get(f"{st['channel']}:{st['thread_ts']}") is not None:
            elems.append(button(":robot_face: Ask Claude to look at it", "sac_job_analyze", jid, "primary"))
        await self.hub.post(st["channel"], st["thread_ts"], f"Job {meta.get('name')} {state}", blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": body[:2900]}},
            {"type": "actions", "block_id": f"sac_jobdone_{jid}", "elements": elems}])
        self.hub.audit.write(event="job-done", job=jid, state=state, exit_code=status.get("exit_code"),
                             session=meta.get("session_id"))

    async def _alerts(self, jid: str, st: dict[str, Any], meta: dict[str, Any]) -> None:
        new = await asyncio.to_thread(self._new_log, jid, st)
        if not new:
            return
        seen = set(st.get("alerts") or [])
        for kind, rx in ALERTS:
            if kind in seen:
                continue
            m = rx.search(new)
            if not m:
                continue
            seen.add(kind)
            line = new[max(0, new.rfind("\n", 0, m.start()) + 1):].split("\n", 1)[0]
            await self.hub.post(st["channel"], st["thread_ts"],
                                f"<@{st['owner']}> :rotating_light: Job *{escape(str(meta.get('name')))}* (`{jid}`) "
                                f"log shows {inline_code(policy.redact_log(line)[:200])} — it keeps running; "
                                "tap Cancel on the job message to stop it.")
        st["alerts"] = sorted(seen)

    # -- main pass ----------------------------------------------------------------
    async def scan(self) -> None:
        jobs = await asyncio.to_thread(self._snapshot)
        changed = False
        now = time.time()
        for jid, meta, status in jobs:
            state = str(status.get("state", ""))
            st = self.state.get(jid)
            if st is None:
                if self._first_run and state in _DONE:
                    self.state[jid] = {"done": True}  # jobs from before the bot tracked them
                    changed = True
                    continue
                dest = await self._destination(jid, meta)
                if dest is None:
                    continue
                st = self.state[jid] = {"channel": dest[0], "thread_ts": dest[1], "owner": dest[2],
                                        "seen_state": None, "progress_at": 0.0}
                changed = True
            if st.get("done"):
                continue
            if state != st.get("seen_state"):
                st["seen_state"] = state
                st["progress_at"] = now
                await self._update(jid, st, meta, status)
                if state == "running" or state in _DONE:
                    await self._alerts(jid, st, meta)
                if state in _DONE:
                    await self._finished(jid, st, meta, status)
                    st["done"] = True
                changed = True
            elif state == "running":
                await self._alerts(jid, st, meta)
                every = max(1, int(meta.get("progress_min") or 30)) * 60
                if now - float(st.get("progress_at") or 0) >= every:
                    st["progress_at"] = now
                    await self._update(jid, st, meta, status)
                changed = True
        if changed or self._first_run:
            self._first_run = False
            self.save()

    # -- buttons and commands -------------------------------------------------------------
    def describe(self, jid: str) -> str:
        return str(self.cli.describe(jid))

    def recent(self, n: int = 15) -> list[str]:
        jobs = self.cli.all_jobs()
        active = [j for j in jobs if self.cli.read_status(j).get("state") in ("queued", "running")]
        rest = [j for j in jobs if j not in active]
        return active + rest[-max(0, n - len(active)):][::-1]

    def cancel(self, jid: str, user: str) -> str:
        r = subprocess.run([sys.executable, "-I", str(CLI), "cancel", jid], capture_output=True, text=True,
                           timeout=120, env={"SAC_JOBS_DIR": str(self.dir), "PATH": "/usr/bin:/bin",
                                             "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"})
        self.hub.audit.write(event="job-cancel", job=jid, by=user, ok=r.returncode == 0)
        return (r.stdout or r.stderr).strip()[:300]

    async def on_action(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        user = body.get("user", {}).get("id", "")
        if user not in self.cfg.allowed_users:
            log.warning("job click from non-allowlisted user %s ignored", user)
            return
        jid = str(action.get("value", ""))
        if not re.fullmatch(r"[0-9a-f]{6}", jid) or not (self.dir / jid).is_dir():
            return
        st = self.state.get(jid) or {}
        channel = st.get("channel") or body.get("channel", {}).get("id")
        thread = st.get("thread_ts") or body.get("message", {}).get("thread_ts")
        if not channel:
            return
        aid = re.sub(r"_h[0-9a-f]{6}$", "", str(action.get("action_id")))  # Home tab buttons carry the id
        if aid == "sac_job_log":
            t = self.tail(jid, 40) or "(no output yet)"
            await self.hub.post(channel, thread, f"*Log of `{jid}`* (last 40 lines)\n```{escape(t[-3500:])}```")
        elif aid == "sac_job_cancel":
            out = await asyncio.to_thread(self.cancel, jid, user)
            await self.hub.post(channel, thread, f":stop_button: {escape(out) or 'cancel requested'} (by <@{user}>)")
            await self.scan()
        elif aid == "sac_job_full":
            err = await self.hub.upload_file(channel, thread, self._log_path(jid), user)
            if err:
                await self.hub.post(channel, thread, f":warning: {escape(err)}")
        elif aid == "sac_job_analyze":
            meta, status = self.cli.read_meta(jid), self.cli.read_status(jid)
            prompt = (f"The job {meta.get('name')} ({jid}) has {status.get('state')} "
                      f"(exit code {status.get('exit_code')}). Command: {' '.join(meta.get('command') or [])} in "
                      f"{meta.get('cwd')}. Its full output is in {self._log_path(jid)}. Look at the output and the "
                      "result files, then summarize what happened and what to do next.")
            await self.front._prompt(channel, str(time.time()), thread, user, prompt)
