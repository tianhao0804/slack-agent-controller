"""Terminal mirror: Claude Code hook events -> Slack threads.

Terminal sessions (``claude`` run by the user on this node, in MIRROR_DIRS)
report hook events through ``deploy/sac_hook.py`` to a user-private Unix
socket. For each session the bot keeps one Slack thread:

* a live "running" message per turn (prompt + one line per tool call),
* when the turn ends, that message becomes a summary (prompt, Claude's last
  reply, stats) and the full turn log is attached as ``turn-NNN.md``,
* a lock while the terminal has the session open: Slack prompts in that
  thread are refused until the terminal exits (SessionEnd, or its process is
  gone); afterwards replying in the thread continues the session from Slack.

The hook payload is only used for metadata; content always comes from the
session's transcript file, which must live under ``~/.claude/projects``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import policy
from . import transcript_md as tm
from .config import EFFORTS, MODEL_ALIASES
from .slackfmt import escape, head_text, inline_code, md_to_mrkdwn
from .store import _atomic_write, load_json_with_backup

if TYPE_CHECKING:
    from .hub import Hub

log = logging.getLogger(__name__)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
EVENTS = {"SessionStart", "UserPromptSubmit", "PreToolUse", "Stop", "Notification", "SessionEnd"}
PROGRESS_INTERVAL_S = 2.0
# The Stop hook fires before the CLI has appended the final reply to the
# transcript (verified); wait for the turn's end marker or for the file to settle.
STOP_WAIT_S = 8.0
RETRY_UPLOADS_S = 60.0
STOP_QUIET_S = 1.0
_ASK_TYPES = {"permission_prompt", "elicitation_dialog"}


def _pid_start(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _is_background(pid: int) -> bool:
    """True if ``pid`` is a Claude Code background session (its parent is the
    daemon's ``claude bg-pty-host``): leaving its terminal only detaches it."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        with open(f"/proc/{ppid}/cmdline", "rb") as f:
            return b"bg-pty-host" in f.read()
    except (OSError, ValueError, IndexError):
        return False


def _terminal_choice(entries: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Model alias and effort of the last terminal (CLI) reply in ``entries``."""
    model = effort = None
    for e in entries:
        if e.get("type") == "assistant" and e.get("entrypoint") in (None, "cli") and not e.get("isSidechain"):
            m = str(e.get("requestedModel") or (e.get("message") or {}).get("model") or "").lower()
            model = next((a for a in MODEL_ALIASES if a in m), model)
            effort = e.get("effort") if e.get("effort") in EFFORTS else effort
    return model, effort


def _clock(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t))


def _dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    return f"{s}s" if s < 60 else (f"{s // 60}m{s % 60:02d}s" if s < 3600 else f"{s // 3600}h{s % 3600 // 60:02d}m")


def _k(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.1f}k" if n >= 1e3 else str(n))


@dataclass
class MirrorState:
    """Persisted per terminal session (mirror.json)."""

    session_id: str
    cwd: str
    transcript: str
    channel: str | None = None
    thread_ts: str | None = None
    own_root: bool = False          # the thread root was posted by the mirror (we may edit it)
    offset: int = 0                 # transcript bytes already mirrored
    agent_offsets: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    open: bool = False              # a terminal has the session open
    pid: int | None = None
    pid_start: int | None = None
    opened_at: float = 0.0
    background: bool = False        # Claude Code background session (detaching does not end it)
    announced_pid: int | None = None  # terminal process whose opening was already posted / handled
    pending_uploads: list[dict[str, Any]] = field(default_factory=list)  # turn logs not yet in Slack
    muted: bool = False             # thread root deleted: stop mirroring until the session is reopened


@dataclass
class _Live:
    """Runtime-only state of the turn in progress."""

    progress_ts: str | None = None
    prompts: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    n_tools: int = 0
    started: float = field(default_factory=time.time)
    last_update: float = 0.0
    flush: asyncio.Task[None] | None = None


class Mirror:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.cfg = hub.cfg
        self.path = self.cfg.state_dir / "mirror.json"
        self.states: dict[str, MirrorState] = {}
        self.live: dict[str, _Live] = {}
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10_000)
        self._server: asyncio.AbstractServer | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._retrier: asyncio.Task[None] | None = None
        self._ignored: set[str] = set()
        self._projects = (Path.home() / ".claude" / "projects").resolve()
        raw = load_json_with_backup(self.path)
        for sid, v in (raw.items() if isinstance(raw, dict) else []):
            try:
                self.states[sid] = MirrorState(**v)
            except TypeError:
                log.warning("skipping malformed mirror state %s", sid)

    def save(self) -> None:
        _atomic_write(self.path, json.dumps({k: asdict(v) for k, v in self.states.items()}, indent=1), backup=True)

    # -- socket server -----------------------------------------------------
    async def start(self) -> None:
        sock = self.cfg.mirror_socket
        assert sock is not None
        sock.parent.mkdir(parents=True, exist_ok=True)
        if sock.is_socket():
            sock.unlink()
        old = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._client, path=str(sock), limit=1 << 18)
        finally:
            os.umask(old)
        os.chmod(sock, 0o600)
        self._consumer = asyncio.create_task(self._consume(), name="mirror-consumer")
        self._retrier = asyncio.create_task(self._retry_loop(), name="mirror-upload-retry")
        log.info("terminal mirror listening on %s for %s", sock, ",".join(map(str, self.cfg.mirror_dirs)))

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        if self._retrier:
            self._retrier.cancel()
        if self._consumer:
            self._consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._consumer
        for lv in self.live.values():
            if lv.flush:
                lv.flush.cancel()
        with contextlib.suppress(OSError):
            if self.cfg.mirror_socket and self.cfg.mirror_socket.is_socket():
                self.cfg.mirror_socket.unlink()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            s: socket.socket = writer.get_extra_info("socket")
            _pid, uid, _gid = struct.unpack("3i", s.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                                struct.calcsize("3i")))
            if uid != os.getuid():
                log.warning("mirror: rejected connection from uid %s", uid)
                return
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            ev = json.loads(line) if line else None
            if isinstance(ev, dict):
                try:
                    self._lock_now(ev)
                except Exception:
                    log.exception("mirror: fast lock update failed")
                self.queue.put_nowait(ev)
        except (asyncio.QueueFull, TimeoutError, ValueError, OSError, asyncio.LimitOverrunError) as e:
            log.warning("mirror: dropped event (%s)", type(e).__name__)
        finally:
            writer.close()

    async def _consume(self) -> None:
        while True:
            ev = await self.queue.get()
            try:
                await self.handle(ev)
            except Exception:
                log.exception("mirror: failed to handle %s", ev.get("hook_event_name"))

    # -- lock --------------------------------------------------------------
    def locked(self, session_id: str) -> MirrorState | None:
        """The state if a terminal currently has this session open."""
        st = self.states.get(session_id)
        if st is None or not st.open:
            return None
        if st.pid and _pid_start(st.pid) != st.pid_start:
            log.info("mirror: terminal process %s for %s is gone; unlocking", st.pid, session_id)
            st.open = False
            self.save()
            return None
        return st

    def lock_text(self, st: MirrorState) -> str:
        sid8 = st.session_id[:8]
        head = (f":lock: This session is open in a terminal on {self.cfg.node_name} (pid {st.pid or '?'}, since "
                f"{_clock(st.opened_at)}), so this message was not sent to Claude.")
        if st.background:
            return head + (f" It runs as a Claude Code *background session*: leaving its terminal only detaches it. "
                           f"On {self.cfg.node_name} run `claude stop {sid8}` (ends it, the conversation is kept), then send your "
                           f"message here again; or `claude attach {sid8}` to go back to it. `!fork` here branches "
                           "off a copy instead.")
        return head + " Exit it in the terminal first (`/exit`), or `!fork` here to branch off a copy."

    def forget_thread(self, session_id: str) -> None:
        """The session's thread root was deleted: stop mirroring this terminal
        session (it is mirrored again, into a new thread, when reopened)."""
        st = self.states.get(session_id)
        if st is not None:
            st.muted, st.thread_ts, st.own_root = True, None, False
            self.live.pop(session_id, None)
            self.save()

    def unlock(self, session_id: str) -> bool:
        st = self.states.get(session_id)
        if st is None or not st.open:
            return False
        st.open = False
        self.save()
        return True

    # -- events ------------------------------------------------------------
    def _accept(self, ev: dict[str, Any]) -> tuple[str, Path, Path] | None:
        name, sid = ev.get("hook_event_name"), str(ev.get("session_id", ""))
        if name not in EVENTS or not _UUID.match(sid) or sid in self._ignored:
            return None
        st = self.states.get(sid)
        tp = Path(str(ev.get("transcript_path") or (st.transcript if st else "")))
        try:
            real_tp = tp.resolve()
        except OSError:
            return None
        if not real_tp.is_relative_to(self._projects) or real_tp.name != f"{sid}.jsonl":
            log.warning("mirror: ignoring %s with unexpected transcript path", sid)
            self._ignored.add(sid)
            return None
        if st is not None:
            return sid, real_tp, Path(st.cwd)
        cwd_raw = str(ev.get("cwd") or "")
        cwd, err = self.hub.validate_cwd(cwd_raw) if cwd_raw else (None, "no cwd")
        if cwd is None or not self.cfg.mirrored(Path(os.path.realpath(cwd))):
            log.debug("mirror: not mirroring %s (%s)", sid, err or f"{cwd_raw} not in MIRROR_DIRS")
            self._ignored.add(sid)
            return None
        return sid, real_tp, cwd

    def _state_for(self, ev: dict[str, Any]) -> MirrorState | None:
        acc = self._accept(ev)
        if acc is None:
            return None
        sid, tp, cwd = acc
        st = self.states.get(sid)
        if st is None:
            st = MirrorState(session_id=sid, cwd=str(cwd), transcript=str(tp))
            self._baseline(st)  # only what happens from now on is mirrored
            self.states[sid] = st
        return st

    def _lock_now(self, ev: dict[str, Any]) -> None:
        """Take the terminal lock as soon as an event arrives, ahead of the slow
        queue (transcript waits, uploads): a Slack prompt must not slip in (review R5).
        Unlocking stays in order, after the last turn has been mirrored."""
        pid = ev.get("claude_pid")
        if ev.get("hook_event_name") in (None, "SessionEnd") or not pid:
            return
        st = self._state_for(ev)
        if st is None or (st.open and st.pid == pid):
            return
        st.open, st.pid, st.pid_start = True, int(pid), ev.get("claude_pid_start")
        st.background = _is_background(st.pid)
        st.opened_at = time.time()
        self.save()

    async def handle(self, ev: dict[str, Any]) -> None:
        st = self._state_for(ev)
        if st is None:
            return
        name = ev["hook_event_name"]
        pid = ev.get("claude_pid")
        if name == "SessionStart":
            st.muted = False
            await self._opened(st, ev, source=str(ev.get("source") or ""))
        elif name != "SessionEnd" and pid and (st.announced_pid != pid or not st.open):
            await self._opened(st, ev)  # missed SessionStart (bot was down) or a new process
        if st.muted and name != "SessionEnd":
            self.save()
            return
        if name == "UserPromptSubmit":
            await self._prompt(st, str(ev.get("prompt") or ""))
        elif name == "PreToolUse":
            await self._tool(st, str(ev.get("tool_name") or "?"), ev.get("tool_input") or {})
        elif name == "Stop":
            await self._await_turn_end(st)
            await self._flush_turn(st, str(ev.get("last_assistant_message") or ""))
        elif name == "Notification":
            await self._notify(st, ev)
        elif name == "SessionEnd":
            await self._await_turn_end(st, need_marker=False)
            await self._flush_turn(st)
            await self._closed(st, str(ev.get("reason") or ""))
        self.save()

    async def _await_turn_end(self, st: MirrorState, need_marker: bool = True) -> None:
        """Wait until the transcript holds the turn's end marker (a
        ``turn_duration`` system entry) and stopped growing, at most STOP_WAIT_S."""
        tp = Path(st.transcript)
        t0 = time.monotonic()
        last_size, last_change = -1, t0
        while time.monotonic() - t0 < STOP_WAIT_S:
            size = tp.stat().st_size if tp.exists() else 0
            now = time.monotonic()
            if size != last_size:
                last_size, last_change = size, now
            elif now - last_change >= STOP_QUIET_S:
                if not need_marker:
                    return
                entries, _ = await asyncio.to_thread(tm.read_new, tp, st.offset)
                if any(e.get("type") == "system" and e.get("subtype") == "turn_duration" for e in entries):
                    return
            await asyncio.sleep(0.25)

    def _baseline(self, st: MirrorState) -> None:
        tp = Path(st.transcript)
        st.offset = tp.stat().st_size if tp.exists() else 0
        sub = tp.with_suffix("") / "subagents"
        st.agent_offsets = {f.name: f.stat().st_size for f in sub.glob("agent-*.jsonl")} if sub.is_dir() else {}

    async def _opened(self, st: MirrorState, ev: dict[str, Any], source: str = "") -> None:
        if source == "resume":
            self._baseline(st)  # skip history (and anything Slack wrote) before this resume
        pid = int(ev["claude_pid"]) if ev.get("claude_pid") else None
        fresh = st.announced_pid != pid or not st.open or (pid is None and source in ("startup", "resume", "clear"))
        if st.pid != pid or not st.open:   # not already taken by the fast path
            st.opened_at = time.time()
        st.open, st.pid, st.pid_start = True, pid, ev.get("claude_pid_start")
        st.background = bool(pid and _is_background(pid))
        if not fresh:
            return
        st.announced_pid = pid
        self.hub.audit.write(event="terminal-open", session=st.session_id, pid=st.pid, source=source or None)
        if st.muted or (st.thread_ts is None and self.hub.store.find_by_session(st.session_id) is None):
            return  # a new thread is only created on the first prompt
        await self._ensure_thread(st)
        await self._post(st, f":lock: Opened in a terminal on {self.cfg.node_name} — mirrored here. Slack replies in this thread "
                             "are paused until the terminal exits.")
        await self._release_bot_process(st)

    async def _release_bot_process(self, st: MirrorState) -> None:
        """The bot's own CLI for this session would hold a stale history."""
        rec = self.hub.store.find_by_session(st.session_id)
        sess = self.hub.sessions.get(rec.key) if rec else None
        if sess is None or not sess.connected:
            return
        if sess.busy:
            await self._post(st, ":warning: A Slack turn is still running in this session. Two writers on one "
                                 "session can corrupt its history — let it finish or send `!stop` here.")
        elif sess.bg_tasks:
            await self._post(st, ":warning: The Slack side still has background task(s) running in this session; "
                                 "they keep running, but avoid working in the terminal until they finish.")
        else:
            await sess.close()

    async def _closed(self, st: MirrorState, reason: str) -> None:
        if not st.open:
            return
        st.open, st.announced_pid = False, None
        self.hub.audit.write(event="terminal-close", session=st.session_id, reason=reason or None)
        if st.thread_ts is None:
            return
        shown = "" if reason in ("", "other") else f" ({escape(reason)})"
        await self._post(st, f":unlock: Terminal closed{shown}. "
                             "Reply in this thread to continue the session from Slack.")
        if st.own_root:
            await self._update(st, st.thread_ts, self._root_text(st))

    async def _ensure_thread(self, st: MirrorState) -> bool:
        if st.muted:
            return False
        rec = self.hub.store.find_by_session(st.session_id)
        if st.thread_ts and rec is not None and (rec.channel, rec.thread_ts) == (st.channel, st.thread_ts):
            return True
        st.thread_ts, st.own_root = None, False
        if rec is not None:
            st.channel, st.thread_ts = rec.channel, rec.thread_ts
            return True
        channel = self.cfg.mirror_channel
        assert channel and self.cfg.mirror_user
        st.channel = channel
        ts = await self.hub.post(channel, None, self._root_text(st))
        if ts is None:
            return False
        st.thread_ts, st.own_root = ts, True
        self.hub.create_session(channel, ts, Path(st.cwd), self.cfg.mirror_user, session_id=st.session_id)
        self.hub.audit.write(event="mirror-thread", session=st.session_id, thread=f"{channel}:{ts}")
        return True

    def _root_text(self, st: MirrorState) -> str:
        head = (f":computer: *Terminal session* in {inline_code(st.cwd)} · `{st.session_id[:8]}`\n"
                f"_Mirrored from a terminal on {self.cfg.node_name}: one message per turn, with the full log attached._\n")
        if st.open:
            return head + ":lock: *Open in the terminal* — Slack replies are paused until it exits."
        return head + ":unlock: Terminal closed — reply in this thread to continue from Slack."

    async def _post(self, st: MirrorState, text: str) -> str | None:
        assert st.channel and st.thread_ts
        return await self.hub.post(st.channel, st.thread_ts, text)

    async def _update(self, st: MirrorState, ts: str, text: str) -> None:
        assert st.channel
        try:
            await self.hub.web.chat_update(channel=st.channel, ts=ts, text=text)
        except Exception:
            log.exception("mirror: chat.update failed")

    # -- turn in progress --------------------------------------------------
    def _live(self, st: MirrorState) -> _Live:
        return self.live.setdefault(st.session_id, _Live())

    async def _prompt(self, st: MirrorState, prompt: str) -> None:
        if not await self._ensure_thread(st):
            return
        lv = self._live(st)
        lv.prompts.append(policy.redact_log(prompt))
        if lv.progress_ts is None:
            lv.started = time.time()
            lv.progress_ts = await self._post(st, self._progress_text(st, lv))
            lv.last_update = time.monotonic()
        else:
            self._schedule(st, lv)

    async def _tool(self, st: MirrorState, tool: str, tin: Any) -> None:
        if not await self._ensure_thread(st):
            return
        lv = self._live(st)
        line = policy.tool_line(tool, tin if isinstance(tin, dict) else {}, Path(st.cwd))
        lv.tools.append(policy.redact_log(line.splitlines()[0] if line else tool))
        lv.n_tools += 1
        if lv.progress_ts is None:
            lv.progress_ts = await self._post(st, self._progress_text(st, lv))
            lv.last_update = time.monotonic()
        else:
            self._schedule(st, lv)

    def _progress_text(self, st: MirrorState, lv: _Live) -> str:
        parts = [f":hourglass_flowing_sand: *Terminal turn {st.turns + 1}* — running since {_clock(lv.started)}"]
        for p in lv.prompts[-3:]:
            parts.append("\n".join("> " + ln for ln in escape(p[:400]).splitlines()[:6]))
        for t in lv.tools[-8:]:
            parts.append(":hammer_and_wrench: " + inline_code(t[:110]))
        if lv.n_tools > 8:
            parts.append(f"_{lv.n_tools} tool calls so far_")
        return "\n".join(parts)

    def _schedule(self, st: MirrorState, lv: _Live) -> None:
        if lv.flush and not lv.flush.done():
            return

        async def later() -> None:
            await asyncio.sleep(max(0.0, lv.last_update + PROGRESS_INTERVAL_S - time.monotonic()))
            if lv.progress_ts:
                lv.last_update = time.monotonic()
                await self._update(st, lv.progress_ts, self._progress_text(st, lv))

        lv.flush = asyncio.create_task(later())

    # -- turn end ----------------------------------------------------------
    # -- attachments outbox -------------------------------------------------
    @property
    def _outbox(self) -> Path:
        d = self.cfg.state_dir / "mirror_outbox"
        d.mkdir(mode=0o700, exist_ok=True)
        return d

    def _queue_upload(self, st: MirrorState, n: int, markdown: str) -> None:
        f = self._outbox / f"{st.session_id}-turn-{n:03d}.md"
        _atomic_write(f, markdown)
        st.pending_uploads.append({"path": str(f), "filename": f"turn-{n:03d}.md", "title": f"Turn {n} — full log",
                                   "tries": 0, "warned": False})
        self.save()

    async def _send_pending(self, st: MirrorState) -> None:
        """Upload queued turn logs in order; stop at the first failure (keeps order)."""
        while st.pending_uploads and st.channel and st.thread_ts and not st.muted:
            item = st.pending_uploads[0]
            path = Path(item["path"])
            try:
                content = path.read_text()
            except OSError:
                log.warning("mirror: queued log %s is gone; skipping", path.name)
                st.pending_uploads.pop(0)
                continue
            try:
                await self.hub.web.files_upload_v2(channel=st.channel, thread_ts=st.thread_ts, content=content,
                                                   filename=item["filename"], title=item["title"])
            except Exception:
                item["tries"] = int(item.get("tries", 0)) + 1
                log.exception("mirror: upload of %s failed (try %d); will retry", item["filename"], item["tries"])
                if not item.get("warned"):
                    item["warned"] = True
                    await self._post(st, f":warning: could not attach `{item['filename']}` yet; the bot keeps it and "
                                         "retries automatically.")
                self.save()
                return
            st.pending_uploads.pop(0)
            with contextlib.suppress(OSError):
                path.unlink()
            if item.get("warned"):
                await self._post(st, f":white_check_mark: attached `{item['filename']}` (delayed).")
            self.save()

    async def _retry_loop(self) -> None:
        while True:
            await asyncio.sleep(RETRY_UPLOADS_S)
            for st in list(self.states.values()):
                if st.pending_uploads:
                    try:
                        await self._send_pending(st)
                    except Exception:
                        log.exception("mirror: retrying uploads failed")

    def _subagent_chunks(self, st: MirrorState) -> tuple[list[tm.SubagentChunk], dict[str, int]]:
        sub = Path(st.transcript).with_suffix("") / "subagents"
        chunks: list[tm.SubagentChunk] = []
        offsets = dict(st.agent_offsets)
        if not sub.is_dir():
            return chunks, offsets
        for f in sorted(sub.glob("agent-*.jsonl"), key=lambda p: p.stat().st_mtime):
            entries, new = tm.read_new(f, offsets.get(f.name, 0))
            if not entries:
                continue
            offsets[f.name] = new
            meta: dict[str, Any] = {}
            with contextlib.suppress(OSError, ValueError):
                meta = json.loads(f.with_name(f.stem + ".meta.json").read_text())
            chunks.append(tm.SubagentChunk(agent_id=f.stem.removeprefix("agent-"),
                                           description=str(meta.get("description") or ""),
                                           tool_use_id=meta.get("toolUseId"), finished=bool(meta.get("finished")),
                                           entries=entries))
        return chunks, offsets

    async def _flush_turn(self, st: MirrorState, last_message: str = "") -> None:
        lv = self.live.pop(st.session_id, None)
        if lv and lv.flush:
            lv.flush.cancel()
        entries, new_offset = await asyncio.to_thread(tm.read_new, Path(st.transcript), st.offset)
        chunks, agent_offsets = await asyncio.to_thread(self._subagent_chunks, st)
        n = st.turns + 1
        title = f"Terminal turn {n} — session {st.session_id}"
        r = await asyncio.to_thread(tm.render_capped, self.cfg, Path(st.cwd), title, entries, chunks)
        if not r.final_text.strip() and last_message.strip():
            r.final_text = policy.redact_log(last_message)
        if r.empty:
            st.offset, st.agent_offsets = new_offset, agent_offsets
            if lv and lv.progress_ts:
                await self._update(st, lv.progress_ts, f":white_check_mark: Terminal turn {n} — nothing new to log")
            return
        if not await self._ensure_thread(st):
            return
        summary = self._summary_text(n, r)
        if lv and lv.progress_ts:
            await self._update(st, lv.progress_ts, summary)
        else:
            await self._post(st, summary)
        # Save first, then upload; a failed upload is retried later instead of being lost (review R4).
        self._queue_upload(st, n, r.markdown)
        await self._send_pending(st)
        st.turns = n
        st.offset, st.agent_offsets = new_offset, agent_offsets
        rec = self.hub.store.find_by_session(st.session_id)
        if rec is not None:
            rec.last_active, rec.has_history = time.time(), True
            # Continue from Slack with what the terminal last used.
            model, effort = _terminal_choice(entries)
            if model:
                rec.model = model
            if effort:
                rec.effort = effort
            self.hub.store.put(rec)
        self.hub.audit.write(event="mirror-turn", session=st.session_id, turn=n, tool_calls=r.tool_calls,
                             subagents=r.subagents, bytes=len(r.markdown.encode()))

    def _summary_text(self, n: int, r: tm.Rendered) -> str:
        stats = [_dur(r.duration_s), f"{r.tool_calls} tool call{'' if r.tool_calls == 1 else 's'}"]
        if r.models:
            stats.append(", ".join(r.models))
        if r.subagents:
            stats.append(f"{r.subagents} sub-agent(s)")
        if r.tokens_in or r.tokens_out:
            stats.append(f"in {_k(r.tokens_in)} / out {_k(r.tokens_out)} tokens")
        parts = [f":computer: *Terminal turn {n}* · " + " · ".join(stats)]
        for p in r.prompts[-3:]:
            parts.append("\n".join("> " + ln for ln in escape(p[:600]).splitlines()[:10]))
        if r.final_text.strip():
            parts.append("*Claude:*\n" + head_text(md_to_mrkdwn(r.final_text.strip()), 2600))
        parts.append(f"_Full log (prompts, thinking, every tool call and output): `turn-{n:03d}.md` below._")
        return head_text("\n".join(parts), 3900)

    async def _notify(self, st: MirrorState, ev: dict[str, Any]) -> None:
        kind = str(ev.get("notification_type") or "")
        msg = str(ev.get("message") or "")
        if not (kind in _ASK_TYPES or (not kind and "permission" in msg.lower())):
            return
        if not await self._ensure_thread(st):
            return
        await self._post(st, f"<@{self.cfg.mirror_user}> :bell: The terminal is waiting for you: "
                             f"{escape(policy.redact_log(msg)[:300])} _(answer it in the terminal)_")
