"""Claude sessions bound to Slack threads, streaming renderer, approvals.

Design notes
------------
* One Slack thread == one ``ThreadSession`` == one ``ClaudeSDKClient``.
* The SDK requires connect()/disconnect() to happen in the same task, so each
  session owns a long-lived worker task that pulls prompts from a queue and
  disconnects itself after IDLE_DISCONNECT_SEC. The next prompt reconnects
  with ``resume=<session_id>``, which is also how sessions survive restarts.
* ``interrupt()`` is called from other tasks (``!stop``); this was verified to
  work against SDK 0.2.182.
* Permissions: a PreToolUse hook applies ``policy.decide`` to every tool call;
  "ask" decisions reach ``can_use_tool``, which posts Slack buttons and waits
  on an ``asyncio.Future`` (auto-deny on timeout).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    get_session_info,
)

from . import policy
from .slackfmt import (
    MSG_LIMIT,
    button,
    escape,
    head_text,
    inline_code,
    md_to_mrkdwn,
    tail_text,
    text_blocks,
)
from .store import SessionRecord

if TYPE_CHECKING:
    from .hub import Hub

log = logging.getLogger(__name__)

PermissionResult = PermissionResultAllow | PermissionResultDeny
Choice = Literal["approve", "remember", "approve_all", "deny", "timeout", "cancel"]

UPLOAD_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".csv", ".tsv", ".txt", ".log",
    ".json", ".html", ".md", ".out", ".dat", ".xyz", ".cif", ".vasp", ".npy", ".npz",
}
_PATH_RE = re.compile(r"(?:~|\.{1,2}|/)?[\w.\-/~]*?[\w\-]+\.(?:" + "|".join(e[1:] for e in UPLOAD_EXTS) + r")\b")

SLACK_SYSTEM_NOTE = """\
# Remote session via Slack
You are being driven remotely from Slack, usually on a phone.
- Keep replies concise and skimmable; avoid huge tables and long code dumps unless asked.
- Every tool call with side effects (Bash, Edit, Write, ...) needs the user's approval through a
  Slack button. If a call is denied, do not retry the same action; ask how to proceed.
- Secrets and config (~/.claude, ~/.aws, ~/.ssh, the bot's own config/state) are off-limits; do
  not try to read them or print environment variables.
- When you produce figures or result files, mention their full paths so the user can attach them
  to Slack with `!upload <path>`.
- When you need the user to choose between options, use the AskUserQuestion tool: the user answers
  with buttons in Slack (or a short reply). Keep questions few and options short.
"""


JOBS_NOTE = """\
- Run anything that takes more than a few minutes, or needs GPUs, as a tracked job instead of a
  background command (those stop when this session's process restarts):
  `{sac_job} submit --name <short-name> --gpus <N> -- <command> [args]`
  (N=0 for CPU jobs; GPUs are picked automatically and the job waits in a queue until enough are
  free). The job keeps running on its own; the bot posts progress and the result in this thread, so
  do not poll it or sleep waiting for it. `{sac_job} status <id>` / `log <id>` show its state.
"""

SAC_JOB = Path(__file__).resolve().parent.parent / "deploy" / "sac-job"


@dataclass
class Prompt:
    text: str
    user: str
    task_id: str | None = None  # record in hub.tasks (survives a bot restart)


@dataclass
class PendingApproval:
    id: str
    session: ThreadSession
    tool: str
    future: asyncio.Future[tuple[Choice, str | None]]
    channel: str
    ts: str | None = None
    can_remember: bool = False
    line: str = ""  # one-line summary of the request (redacted), for the Home tab


def _fmt_dur(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


FINAL_RETRY_DELAYS = (1.0, 3.0, 9.0)
INTERRUPT_TIMEOUT_S = 10.0
STOP_WAIT_S = 15.0


@dataclass
class StopReport:
    was_running: bool       # a reply was running when !stop arrived
    stopped: bool           # ... and it ended within the wait
    error: str | None       # the interrupt request itself failed
    dropped: int            # queued prompts removed
    bg_left: int            # background tasks still running


class TurnRenderer:
    """Accumulates one turn's output and mirrors it into a single Slack
    message via throttled ``chat.update`` calls."""

    def __init__(self, hub: Hub, channel: str, thread_ts: str, cwd: Path) -> None:
        self.hub = hub
        self.channel = channel
        self.thread_ts = thread_ts
        self.cwd = cwd
        self.parts: list[dict[str, Any]] = []  # {"kind": "text"|"tool"|"note", ...}
        self.tools: dict[str, dict[str, Any]] = {}
        self.live = ""
        self.status = "⏳ working…"
        self.ts: str | None = None
        self._dirty = False
        self._last_sent = ""
        self._loop_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        # Buttons under the reply: controls(final) -> Block Kit elements (set by the session).
        self.controls: Callable[[bool], list[dict[str, Any]]] | None = None

    # -- accumulation ------------------------------------------------------
    def delta(self, text: str) -> None:
        self.live += text
        self._dirty = True

    def commit_text(self, text: str) -> None:
        self.live = ""
        if text.strip():
            self.parts.append({"kind": "text", "text": text})
        self._dirty = True

    def tool(self, tool_use_id: str, line: str, sub: bool) -> None:
        if self.live.strip():  # text streamed before a tool call
            self.parts.append({"kind": "text", "text": self.live})
            self.live = ""
        part = {"kind": "tool", "line": line, "sub": sub, "err": False}
        self.parts.append(part)
        self.tools[tool_use_id] = part
        self._dirty = True

    def tool_error(self, tool_use_id: str) -> None:
        if part := self.tools.get(tool_use_id):
            part["err"] = True
            self._dirty = True

    def note(self, text: str) -> None:
        self.parts.append({"kind": "note", "text": text})
        self._dirty = True

    def set_status(self, status: str) -> None:
        self.status = status
        self._dirty = True

    # -- rendering ---------------------------------------------------------
    def render(self) -> str:
        out: list[str] = []
        for p in self.parts:
            if p["kind"] == "text":
                out.append(md_to_mrkdwn(p["text"]).strip())
            elif p["kind"] == "tool":
                mark = " :x:" if p["err"] else ""
                indent = "      ↳ " if p["sub"] else ""
                out.append(f"{indent}:hammer_and_wrench: {inline_code(p['line'])}{mark}")
            else:
                out.append(f"_{escape(p['text'])}_")
        if self.live.strip():
            out.append(md_to_mrkdwn(self.live).strip())
        # Whole text each time, so a token split across stream chunks is still caught (review R7).
        return policy.redact_log("\n".join(x for x in out if x))

    def raw_markdown(self) -> str:
        out: list[str] = []
        for p in self.parts:
            if p["kind"] == "text":
                out.append(p["text"])
            elif p["kind"] == "tool":
                out.append(f"> [tool] {p['line']}{' (error)' if p['err'] else ''}")
            else:
                out.append(f"> {p['text']}")
        return policy.redact_log("\n\n".join(out))

    def final_text(self) -> str:
        return policy.redact_log("\n\n".join(p["text"] for p in self.parts if p["kind"] == "text"))

    # -- Slack I/O ---------------------------------------------------------
    async def start(self) -> None:
        resp = await self.hub.web.chat_postMessage(
            channel=self.channel, thread_ts=self.thread_ts, text=self.status
        )
        self.ts = resp["ts"]
        self._loop_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.hub.cfg.update_interval_s)
                if self._dirty:
                    await self._update(final=False)
        except asyncio.CancelledError:
            pass

    def _compose(self, footer: str | None) -> str:
        body = self.render()
        foot = footer if footer is not None else f"_{self.status}  ·  `!stop` to interrupt_"
        # Keep the end: on a phone the conclusion matters most; the full text is attached when cut.
        body = tail_text(body, MSG_LIMIT - len(foot) - 4)
        return f"{body}\n\n{foot}" if body else foot

    def _blocks(self, text: str, final: bool) -> list[dict[str, Any]]:
        blocks = text_blocks(text, block_id="sac_reply")
        elems = self.controls(final) if self.controls else []
        if elems:
            blocks.append({"type": "actions", "block_id": "sac_reply_ctl", "elements": elems[:5]})
        return blocks

    async def _update(self, final: bool, footer: str | None = None) -> bool:
        """Send the current text. Returns False if Slack rejected it (it stays dirty)."""
        async with self._lock:
            self._dirty = False
            text = self._compose(footer)
            blocks = self._blocks(text, final)
            sig = text + repr([e.get("action_id") for b in blocks if b["type"] == "actions" for e in b["elements"]])
            if sig == self._last_sent or not self.ts:
                return True
            try:
                await self.hub.web.chat_update(channel=self.channel, ts=self.ts, text=text, blocks=blocks)
                self._last_sent = sig
                return True
            except Exception:
                log.exception("chat.update failed")
                self._dirty = True  # the next flush (or finish) tries again
                return False

    async def finish(self, footer: str) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        if self.live.strip():
            self.parts.append({"kind": "text", "text": self.live})
            self.live = ""
        full = self.render()
        too_long = len(full) > MSG_LIMIT - len(footer) - 4
        if too_long:
            footer = footer + "  ·  full output attached :paperclip:"
        # The final text must arrive (review R4): retry, then fall back to a new message.
        for delay in FINAL_RETRY_DELAYS:
            if await self._update(final=True, footer=footer):
                break
            await asyncio.sleep(delay)
        else:
            if not await self._update(final=True, footer=footer):
                try:
                    text = self._compose(footer)
                    await self.hub.web.chat_postMessage(channel=self.channel, thread_ts=self.thread_ts,
                                                        text=text, blocks=self._blocks(text, True))
                    log.warning("final chat.update kept failing; posted the result as a new message")
                except Exception:
                    log.exception("could not deliver the final result to Slack")
        if too_long:
            await self.hub.upload_content(
                self.channel, self.thread_ts, self.raw_markdown(), "claude-output.md", "Full output"
            )
            last = next((p["text"] for p in reversed(self.parts) if p["kind"] == "text" and p["text"].strip()), "")
            if last.strip():
                # On a phone the conclusion is what matters: give it its own clean message.
                body = head_text(md_to_mrkdwn(policy.redact_log(last.strip())), MSG_LIMIT - 40)
                await self.hub.post(self.channel, self.thread_ts, ":pushpin: *Final answer*\n" + body)


@dataclass
class _Turn:
    """One model turn: either answering a Slack prompt or Claude-initiated
    (background task finished, scheduled wake-up, ...)."""

    renderer: TurnRenderer
    owner: str
    started: float
    autonomous: bool
    result: ResultMessage | None = None
    error: str | None = None


# Messages that mean "a turn is producing output" (used to detect turns the
# CLI starts on its own, e.g. after a background task completes).
def _starts_turn(msg: Any) -> bool:
    if isinstance(msg, (StreamEvent, AssistantMessage, ResultMessage)):
        return True
    return isinstance(msg, SystemMessage) and msg.subtype == "init"


class ThreadSession:
    def __init__(self, hub: Hub, rec: SessionRecord) -> None:
        self.hub = hub
        self.rec = rec
        self.queue: asyncio.Queue[Prompt | None] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        self.client: ClaudeSDKClient | None = None
        self.allow_all: set[str] = set()
        self.pending: set[str] = set()
        self.renderer: TurnRenderer | None = None
        self.turn_owner: str = rec.owner
        self.bg_tasks: dict[str, str] = {}  # background task id -> description
        self._turn: _Turn | None = None
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._reader: asyncio.Task[None] | None = None
        self._pending_notes: list[str] = []
        # Transcript size when our last turn ended: if it differs before the next
        # prompt, another Claude Code process (e.g. a terminal) wrote to the session.
        self._baseline: int | None = None
        self._last_activity = time.monotonic()
        self._proc_cost = 0.0  # cumulative cost reported by the current CLI process
        self._options_stale = False  # model/effort changed while a CLI process is running
        self.last_task_id: str | None = None
        self.last_event = time.monotonic()        # last message from the CLI (stall detection)
        self.open_tools: dict[str, str] = {}      # tool_use_id -> one-line summary, until its result arrives
        self.stall_notified = False
        self._call_cwd: dict[str, Path] = {}  # tool_use_id -> directory a Bash call will run in
        self._stderr: collections.deque[str] = collections.deque(maxlen=20)
        self._turn_files: list[str] = []
        self._auth_alerted = False

    @property
    def cfg(self):  # noqa: ANN201 - convenience
        return self.hub.cfg

    @property
    def cwd(self) -> Path:
        return Path(self.rec.cwd)

    @property
    def memory_dir(self) -> Path:
        """The only part of ~/.claude this session may read and write: its project's auto-memory."""
        return policy.project_memory_dir(self.rec.cwd)

    @property
    def busy(self) -> bool:
        return self._turn is not None

    @property
    def connected(self) -> bool:
        return self.client is not None

    # -- public API --------------------------------------------------------
    def submit(self, text: str, user: str) -> int:
        """Queue a prompt; returns the number of prompts ahead of it."""
        ahead = self.queue.qsize() + (1 if self.busy else 0)
        tid = self.hub.tasks.add(self.rec.channel, self.rec.thread_ts, self.rec.session_id, user, text)
        self.last_task_id = tid
        self.queue.put_nowait(Prompt(text, user, tid))
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._worker(), name=f"session-{self.rec.key}")
        return ahead

    def queued_task_ids(self) -> list[str]:
        return [r.id for r in self.hub.tasks.records.values()
                if (r.channel, r.thread_ts, r.session_id) == (self.rec.channel, self.rec.thread_ts, self.rec.session_id)
                and r.status == "queued"]

    def drop_queued(self, task_id: str | None = None) -> int:
        """Remove prompts waiting behind the current one. Returns how many."""
        kept: list[Prompt | None] = []
        dropped: list[Prompt] = []
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if isinstance(item, Prompt) and (task_id is None or item.task_id == task_id):
                dropped.append(item)
            else:
                kept.append(item)  # keep the close sentinel
        for item in kept:
            self.queue.put_nowait(item)
        # A worker may already hold a queued prompt while waiting for a global turn slot.
        waiting = [tid for tid in self.queued_task_ids() if task_id is None or tid == task_id]
        if not self.hub.shutting_down:  # on shutdown they are reported after the restart
            self.hub.tasks.done(*waiting)
        return len(waiting) + sum(p.task_id is None for p in dropped)

    def promote(self, task_id: str) -> bool:
        """Move the queued prompt with ``task_id`` to the front. False if it is not queued."""
        items: list[Prompt | None] = []
        while not self.queue.empty():
            items.append(self.queue.get_nowait())
        hit = [i for i in items if isinstance(i, Prompt) and i.task_id == task_id]
        rest = [i for i in items if not (isinstance(i, Prompt) and i.task_id == task_id)]
        for i in hit + rest:
            self.queue.put_nowait(i)
        return bool(hit)

    async def stop(self, user: str, clear_queue: bool = True, wait_s: float = STOP_WAIT_S) -> StopReport:
        """Interrupt the running reply, deny its pending approvals and (by
        default) drop queued prompts; then wait up to ``wait_s`` for it to end."""
        dropped = self.drop_queued() if clear_queue else 0
        if not self.busy:
            return StopReport(was_running=False, stopped=False, error=None, dropped=dropped,
                              bg_left=len(self.bg_tasks))
        interrupted_turn = self._turn
        for aid in list(self.pending):
            self.hub.resolve_approval(aid, "cancel", user, check_user=False)
        error = None
        if self.client is not None:
            try:
                await asyncio.wait_for(self.client.interrupt(), timeout=INTERRUPT_TIMEOUT_S)
            except TimeoutError:
                error = f"the interrupt request got no answer within {INTERRUPT_TIMEOUT_S:.0f} s"
            except Exception as e:
                log.exception("interrupt failed")
                error = f"interrupt failed: {type(e).__name__}"
        t0 = time.monotonic()
        while self._turn is interrupted_turn and time.monotonic() - t0 < wait_s:
            await asyncio.sleep(0.2)
        self.hub.audit.write(event="stop", session=self.rec.session_id, thread=self.rec.key, by=user,
                             stopped=self._turn is not interrupted_turn, dropped=dropped, error=error)
        return StopReport(was_running=True, stopped=self._turn is not interrupted_turn, error=error, dropped=dropped,
                          bg_left=len(self.bg_tasks))

    async def stop_background(self, user: str) -> tuple[list[str], list[str]]:
        """Ask the CLI to stop every background task. Returns (stopped, failed) descriptions."""
        stopped: list[str] = []
        failed: list[str] = []
        for tid, desc in list(self.bg_tasks.items()):
            if self.client is None:
                failed.append(desc)
                continue
            try:
                await asyncio.wait_for(self.client.stop_task(tid), timeout=INTERRUPT_TIMEOUT_S)
                stopped.append(desc)
            except Exception as e:
                log.warning("stop_task %s failed: %s", tid, type(e).__name__)
                failed.append(desc)
        # The CLI confirms with task_updated(killed); give it a moment so the count is current.
        t0 = time.monotonic()
        while stopped and any(d in self.bg_tasks.values() for d in stopped) and time.monotonic() - t0 < 5:
            await asyncio.sleep(0.2)
        self.hub.audit.write(event="stop-background", session=self.rec.session_id, thread=self.rec.key, by=user,
                             stopped=len(stopped), failed=len(failed))
        return stopped, failed

    async def close(self) -> None:
        """Stop the worker (and its CLI process, including background tasks)."""
        if self.busy:
            await self.stop("system", wait_s=0)
        self.drop_queued()
        self.queue.put_nowait(None)
        if self.worker:
            try:
                await asyncio.wait_for(self.worker, timeout=30)
            except (TimeoutError, Exception):
                self.worker.cancel()

    # -- worker (owns connect/disconnect, as the SDK requires) -------------
    async def _worker(self) -> None:
        try:
            idle = self.cfg.idle_disconnect_s
            while True:
                if self.busy or self.bg_tasks:
                    timeout = idle  # re-check later; activity below resets the clock
                else:
                    timeout = max(idle - (time.monotonic() - self._last_activity), 1.0)
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except TimeoutError:
                    # Never reap a process that is working or has background tasks
                    # running (they would be killed with it), or one that was active
                    # (prompt, Claude-initiated turn, task start/finish) within `idle`.
                    if self.busy or self.bg_tasks or time.monotonic() - self._last_activity < idle:
                        continue
                    log.info("session %s idle; disconnecting", self.rec.session_id)
                    break
                if item is None:
                    break
                await self._run_prompt(item)
        finally:
            await self._disconnect()

    async def _disconnect(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self.client is None:
            return
        client, self.client = self.client, None
        self._options_stale = False
        self.bg_tasks.clear()
        try:
            await client.disconnect()
        except Exception:
            log.exception("disconnect failed")

    def _system_append(self) -> str:
        note = SLACK_SYSTEM_NOTE + (JOBS_NOTE.replace("{sac_job}", str(SAC_JOB)) if self.cfg.jobs_enabled else "")
        parts = [note,
                 f"Your auto-memory directory for this project is {self.memory_dir}; you may read and write "
                 "notes there with Read/Write/Edit (never secrets). The rest of ~/.claude stays off-limits."]
        if self.cfg.load_claude_md:
            # setting_sources=[] disables the CLI's own CLAUDE.md loading (it
            # would also load settings files whose allow-rules bypass Slack
            # approvals), so inject memory files ourselves.
            cands = [Path.home() / ".claude" / "CLAUDE.md"]
            d = self.cwd
            chain: list[Path] = []
            while True:
                chain.append(d)
                if d == d.parent or not self.cfg.under_roots(d.parent):
                    break
                d = d.parent
            for d in reversed(chain):
                cands += [d / "CLAUDE.md", d / ".claude" / "CLAUDE.md"]
            for c in dict.fromkeys(cands):
                try:
                    if c.is_file():
                        parts.append(f"# Contents of {c} (project instructions)\n{c.read_text()[:20000]}")
                except OSError:
                    pass
        return "\n\n".join(parts)

    def _options(self) -> ClaudeAgentOptions:
        # Three cases: continue our own transcript; first turn of a fork (resume
        # the parent with fork_session, under our pre-assigned id); brand new.
        if not self.rec.has_history and self._transcript_file() is not None:
            # A first turn that never reached its result (CLI crash, bot restart) has
            # still written a transcript under our id: resume it (review R3).
            log.info("session %s has a transcript but no finished turn; resuming it", self.rec.session_id)
            self.rec.has_history = True
            self.hub.store.put(self.rec)
        fork = bool(self.rec.fork_from) and not self.rec.has_history
        resume = self.rec.session_id if self.rec.has_history else (self.rec.fork_from if fork else None)
        env = {
            **({"CLAUDE_CODE_USE_BEDROCK": "1"} if self.cfg.provider == "bedrock" else {}),
            "DISABLE_AUTOUPDATER": "1",
            # Needs bubblewrap + socat sandbox; off unless configured.
            "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1" if self.cfg.scrub_subprocess_env else "0",
        }
        return ClaudeAgentOptions(
            cwd=str(self.cwd),
            resume=resume,
            session_id=self.rec.session_id if (fork or not resume) else None,
            fork_session=fork,
            permission_mode="auto" if self.rec.mode == "auto" else "default",
            allowed_tools=[],
            disallowed_tools=list(self.cfg.disallowed_tools),
            can_use_tool=self._can_use_tool,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[self._pre_tool_hook])]},  # type: ignore[list-item]
            setting_sources=[],
            system_prompt={"type": "preset", "preset": "claude_code", "append": self._system_append()},
            env=env,
            # Pin auto-memory to this project's directory, the one place in ~/.claude the policy opens.
            settings=json.dumps({"autoMemoryDirectory": str(self.memory_dir)}),
            include_partial_messages=True,
            model=self.rec.model or self.cfg.model,
            effort=self.rec.effort or self.cfg.default_effort,  # type: ignore[arg-type]
            max_budget_usd=self.cfg.max_budget_usd,
            max_turns=self.cfg.max_turns,
            cli_path=self.cfg.cli_path,
            stderr=self._on_stderr,
            client_app="slack-agent-controller/0.1",
        )

    def _on_stderr(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self._stderr.append(policy.redact(line)[:300])
            log.debug("cli[%s]: %s", self.rec.session_id[:8], self._stderr[-1])

    async def _ensure_client(self) -> None:
        if self.client is not None and self._reader is not None and self._reader.done():
            await self._disconnect()  # CLI exited; start a fresh process (resume)
        if self.client is not None:
            return
        client = ClaudeSDKClient(self._options())
        await client.connect()
        self.client = client
        # A resumed CLI restores the session's cumulative cost from the
        # transcript (verified: a session with $56.25 of terminal use reported
        # $57.35 after one Slack turn), so that is the baseline for deltas.
        self._proc_cost = await asyncio.to_thread(self._restored_cost)
        self._reader = asyncio.create_task(self._read_loop(), name=f"reader-{self.rec.key}")

    def _restored_cost(self) -> float:
        """Last ``cost-state.totalCostUSD`` in the transcript being resumed (0 if none)."""
        sid = self.rec.session_id if self.rec.has_history else self.rec.fork_from
        if not sid:
            return 0.0
        f = self._transcript_file(sid)
        if f is None:
            return 0.0
        total = 0.0
        try:
            with f.open("rb") as fh:
                for line in fh:
                    if b'"cost-state"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("type") == "cost-state":
                        total = float(e.get("totalCostUSD") or 0.0)
        except OSError:
            return 0.0
        return total

    # -- turns -------------------------------------------------------------
    def _begin(self, turn: _Turn) -> None:
        self.last_event = time.monotonic()
        self.stall_notified = False
        self.open_tools.clear()
        self._turn = turn
        self._last_activity = time.monotonic()
        self.renderer = turn.renderer
        self.turn_owner = turn.owner
        self._turn_files = []
        self._turn_done.clear()

    async def _end_turn(self) -> None:
        turn = self._turn
        if turn is None:
            return
        self._turn = None
        self._last_activity = time.monotonic()
        self.renderer = None
        try:
            await self._finish_turn(turn, time.monotonic() - turn.started)
        finally:
            self._turn_done.set()

    async def _transcript_size(self) -> int | None:
        """Size of this session's transcript file. Uses the SDK's metadata
        lookup (file stat); the transcript contents are not read."""
        try:
            info = await asyncio.to_thread(get_session_info, self.rec.session_id)
        except Exception:
            log.exception("transcript size lookup failed")
            return None
        return info.file_size if info else None

    async def _reload_if_changed(self) -> str | None:
        """If the transcript changed since our last turn, another Claude Code
        process (typically a terminal `claude --resume`) added to this session.
        Our CLI process still holds the older history in memory, so drop it: the
        next connect resumes from the transcript and sees everything.
        Returns a note for the thread when a reload happened. Must only be
        called while no turn is running."""
        if self.client is None or self._baseline is None or not self.rec.has_history:
            return None
        size = await self._transcript_size()
        if size is None or size == self._baseline:
            return None
        if self._turn is not None or not self._turn_done.is_set():
            return None  # a Claude-initiated turn started meanwhile; caller waits and retries
        log.info("session %s: transcript changed outside the bot (%s -> %s bytes); reloading",
                 self.rec.session_id, self._baseline, size)
        self.hub.audit.write(event="session-reload", session=self.rec.session_id, thread=self.rec.key,
                             reason="transcript changed outside the bot", bytes_before=self._baseline,
                             bytes_after=size)
        lost = f"; {len(self.bg_tasks)} background task(s) of this Slack session were ended" if self.bg_tasks else ""
        await self._disconnect()
        self._baseline = size
        return f"detected new activity on this session outside Slack (e.g. a terminal); reloaded it from the transcript{lost}"

    async def _run_prompt(self, item: Prompt) -> None:
        # Wait for any Claude-initiated turn to finish first, then make sure our
        # in-memory history is not older than the transcript.
        reload_note: str | None = None
        while True:
            while self._turn is not None or not self._turn_done.is_set():
                await self._turn_done.wait()
            reload_note = await self._reload_if_changed() or reload_note
            if self._options_stale and self.client is not None and not self.bg_tasks:
                await self._disconnect()  # model/effort changed: restart with the new options
            if self._turn is None and self._turn_done.is_set():
                break
        if item.task_id and item.task_id not in self.hub.tasks.records:
            return
        if await self._refuse_if_terminal_locked(item):
            self.hub.tasks.done(item.task_id)
            return
        slots = self.hub.turn_slots
        if slots.locked():
            busy = sum(1 for s in self.hub.sessions.values() if s.busy)
            await self.hub.post(self.rec.channel, self.rec.thread_ts,
                                f"_Waiting: {busy} other session(s) are replying (limit {self.cfg.max_running_turns},"
                                " MAX_RUNNING_TURNS). This starts automatically._")
        async with slots:
            if item.task_id and item.task_id not in self.hub.tasks.records:
                return
            if await self._refuse_if_terminal_locked(item):  # it may have been opened while we waited
                self.hub.tasks.done(item.task_id)
                return
            self.hub.tasks.mark(item.task_id, "running")
            try:
                await self._run_turn(item, reload_note)
            finally:
                if self.hub.shutting_down:
                    self.hub.tasks.mark(item.task_id, "interrupted")
                else:
                    self.hub.tasks.done(item.task_id)

    async def _refuse_if_terminal_locked(self, item: Prompt) -> bool:
        mirror = self.hub.mirror
        lk = mirror.locked(self.rec.session_id) if mirror is not None else None
        if mirror is None or lk is None:
            return False
        # The terminal opened this session while the prompt was queued (review R5).
        await self.hub.offer_fork(self.rec, item.user, item.text, "Your queued message was not sent. "
                                  + mirror.lock_text(lk) + f"\n> {escape(policy.redact_log(item.text)[:300])}")
        self.hub.audit.write(event="queued-prompt-refused", session=self.rec.session_id,
                             thread=self.rec.key, reason="terminal lock", pid=lk.pid)
        return True

    async def _run_turn(self, item: Prompt, reload_note: str | None) -> None:
        turn = _Turn(TurnRenderer(self.hub, self.rec.channel, self.rec.thread_ts, self.cwd),
                     owner=item.user, started=time.monotonic(), autonomous=False)
        turn.renderer.controls = self.controls
        self._begin(turn)
        try:
            await turn.renderer.start()
            if reload_note:
                turn.renderer.note(reload_note)
            await self._ensure_client()
            assert self.client is not None
            await self.client.query(item.text)
        except Exception as e:  # resume failure, CLI crash, Slack error ...
            log.exception("prompt failed in session %s", self.rec.session_id)
            turn.error = self._error_text(e)
            if self._turn is turn:
                await self._end_turn()
            await self._disconnect()
            return
        # The reader finishes the turn when the ResultMessage arrives.
        await self._turn_done.wait()

    def _error_text(self, e: BaseException | None) -> str:
        tail = " | ".join(list(self._stderr)[-3:])
        bg = re.search(r"session is running in the background \((\w+)\)", f"{e} {tail}")
        if bg:
            return (f"this session is still running as a Claude Code background session ({bg.group(1)}), so it "
                    f"cannot be resumed here. On {self.cfg.node_name} run `claude stop {bg.group(1)}` (ends it, the conversation "
                    f"is kept) and send your message again, or `claude attach {bg.group(1)}` to continue there. "
                    "`!fork` here branches off a copy instead.")
        head = f"{type(e).__name__}: {e}"[:500] if e else "Claude Code process exited unexpectedly"
        return head + (f" (cli: {tail})" if tail else "")

    async def _read_loop(self) -> None:
        """Continuously consume CLI output, including turns the CLI starts by
        itself (verified: after a background Bash task completes, the CLI
        injects a turn whose ResultMessage has origin.kind == "task-notification")."""
        assert self.client is not None
        error: BaseException | None = None
        try:
            async for msg in self.client.receive_messages():
                try:
                    await self._route(msg)
                except Exception:
                    log.exception("failed to handle message in session %s", self.rec.session_id)
        except asyncio.CancelledError:
            if self._turn is not None:
                self._turn.error = self._turn.error or "session closed"
                await self._end_turn()
            raise
        except Exception as e:
            log.exception("reader failed in session %s", self.rec.session_id)
            error = e
        # Stream ended: the CLI process is gone.
        self.bg_tasks.clear()
        if self._turn is not None:
            self._turn.error = self._turn.error or self._error_text(error)
            await self._end_turn()

    async def _route(self, msg: Any) -> None:
        self.last_event = time.monotonic()
        self.stall_notified = False
        if isinstance(msg, SystemMessage):
            self._track_background(msg)
        if self._turn is None:
            if not _starts_turn(msg):
                return
            # Claude-initiated turn: render it like any other and notify.
            turn = _Turn(TurnRenderer(self.hub, self.rec.channel, self.rec.thread_ts, self.cwd),
                         owner=self.turn_owner, started=time.monotonic(), autonomous=True)
            turn.renderer.controls = self.controls
            self._begin(turn)
            await turn.renderer.start()
            turn.renderer.note("Claude continued on its own")
            for n in self._pending_notes:
                turn.renderer.note(n)
            self._pending_notes.clear()
        cur = self._turn
        assert cur is not None
        await self._on_message(msg, cur.renderer)
        if isinstance(msg, ResultMessage):
            cur.result = msg
            await self._end_turn()

    def _track_background(self, msg: SystemMessage) -> None:
        d = msg.data or {}
        tid = d.get("task_id")
        if not tid:
            return
        self._last_activity = time.monotonic()
        if msg.subtype == "task_started":
            self.bg_tasks[tid] = str(d.get("description") or tid)
        elif msg.subtype == "task_notification" or (
            msg.subtype == "task_updated" and d.get("status") in ("completed", "failed", "stopped", "killed")
        ):
            desc = self.bg_tasks.pop(tid, None)
            if msg.subtype == "task_notification":
                note = f"background task {d.get('status', 'finished')}: {d.get('summary') or desc or tid}"
                if self._turn is not None:
                    self._turn.renderer.note(note)
                else:
                    self._pending_notes.append(note)

    async def _finish_turn(self, turn: _Turn, elapsed: float) -> None:
        rec = self.rec
        r, result, error = turn.renderer, turn.result, turn.error
        if result is not None:
            cur = result.total_cost_usd or 0.0
            # total_cost_usd is cumulative per CLI process (verified), so add the delta.
            delta = cur - self._proc_cost if cur >= self._proc_cost else cur
            self._proc_cost = cur
            rec.cost_usd += max(0.0, delta)
            u = result.usage or {}
            rec.input_tokens += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) + int(
                u.get("cache_creation_input_tokens", 0)
            )
            rec.output_tokens += int(u.get("output_tokens", 0))
            rec.turns += 1
            rec.has_history = True
            if result.session_id and result.session_id != rec.session_id:
                log.info("session id changed %s -> %s", rec.session_id, result.session_id)
                rec.session_id = result.session_id
            if result.api_error_status in (401, 403):
                await self._auth_alert()
        rec.last_active = time.time()
        self.hub.store.put(rec)
        self._baseline = await self._transcript_size()

        cost = f"${rec.cost_usd:.2f} total"
        if error:
            footer = f":x: *error* after {_fmt_dur(elapsed)}: {escape(error)}"
        elif result is None:
            footer = f":warning: no result after {_fmt_dur(elapsed)}"
        elif (result.terminal_reason or "").startswith("aborted"):
            footer = f":stop_button: interrupted after {_fmt_dur(elapsed)} · {cost}"
        elif result.is_error:
            detail = result.subtype if result.subtype != "success" else (result.errors or ["error"])[0]
            footer = f":warning: ended with {escape(str(detail))} after {_fmt_dur(elapsed)} · {cost}"
        else:
            footer = f":white_check_mark: done in {_fmt_dur(elapsed)} · {result.num_turns} steps · {cost}"
        if turn.autonomous:
            footer = ":robot_face: _Claude-initiated_ · " + footer
        if self.rec.mode == "auto":
            footer += " · :zap: auto mode"
        if self.bg_tasks:
            footer += f" · :gear: {len(self.bg_tasks)} background task(s) running"
        try:
            await r.finish(footer)
        except Exception:
            log.exception("failed to finalize turn message")

        await self._offer_uploads(r.final_text())
        aborted = bool(result and (result.terminal_reason or "").startswith("aborted"))
        failed = bool(error) or result is None or (bool(result.is_error) and not aborted)
        # Always notify for Claude-initiated turns (nobody is watching) and for failures.
        if turn.autonomous or failed or elapsed >= self.cfg.notify_after_s:
            icon = ":x:" if failed else (":robot_face:" if turn.autonomous else ":white_check_mark:")
            what = "failed" if failed else ("Claude reported back" if turn.autonomous else "finished")
            if aborted and not failed:
                icon, what = ":stop_button:", "stopped"
            await self.hub.post(
                rec.channel, rec.thread_ts, f"<@{turn.owner}> {icon} {what} ({_fmt_dur(elapsed)}, {cost})"
            )

    async def _auth_alert(self) -> None:
        if self._auth_alerted:
            return
        self._auth_alerted = True
        await self.hub.post(
            self.rec.channel,
            self.rec.thread_ts,
            f"<@{self.turn_owner}> :rotating_light: Authentication with the model provider ({self.cfg.provider}) "
            f"failed; the API key has probably expired. Update it in `{self.cfg.env_file}` and run "
            "`systemctl --user restart slack-agent-controller`.",
        )

    async def _on_message(self, msg: Any, r: TurnRenderer) -> None:
        if isinstance(msg, StreamEvent):
            if msg.parent_tool_use_id is None:
                ev = msg.event
                if ev.get("type") == "content_block_delta":
                    d = ev.get("delta", {})
                    if d.get("type") == "text_delta":
                        r.delta(d.get("text", ""))
        elif isinstance(msg, AssistantMessage):
            sub = msg.parent_tool_use_id is not None
            if msg.error:
                r.note(f"API error: {msg.error}")
                if msg.error == "authentication_failed":
                    await self._auth_alert()
            for b in msg.content:
                if isinstance(b, TextBlock) and not sub:
                    r.commit_text(b.text)
                elif isinstance(b, ToolUseBlock):
                    r.tool(b.id, policy.tool_line(b.name, b.input, self.cwd), sub)
                    self.open_tools[b.id] = policy.redact_log(policy.tool_line(b.name, b.input, self.cwd))
                    if b.name in ("Write", "NotebookEdit", "Edit"):
                        p = b.input.get("file_path") or b.input.get("notebook_path")
                        if isinstance(p, str):
                            self._turn_files.append(p)
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    self.open_tools.pop(b.tool_use_id, None)
                    if b.is_error:
                        r.tool_error(b.tool_use_id)
        elif isinstance(msg, SystemMessage):
            if msg.subtype == "init":
                sid = msg.data.get("session_id")
                if sid and sid != self.rec.session_id:
                    self.rec.session_id = sid
                    self.hub.store.put(self.rec)
                pm = msg.data.get("permissionMode")
                if self.rec.mode == "auto" and pm and pm != "auto":
                    r.note(f"auto mode is not available for this model; using Slack approvals (mode {pm})")
                model = str(msg.data.get("model", ""))
                if model and self.cfg.provider == "bedrock" and not self.hub.bedrock_model_ok(model):
                    r.note(f"warning: model id {model} does not look like a Bedrock id")
            elif msg.subtype == "compact_boundary":
                r.note("context compacted")

    # -- permissions -------------------------------------------------------
    async def _pre_tool_hook(self, inp: Any, tool_use_id: str | None, ctx: Any) -> dict[str, Any]:
        tool = str(inp.get("tool_name", ""))
        tin = inp.get("tool_input") or {}
        cwd = Path(inp.get("cwd") or self.rec.cwd)
        if tool == "AskUserQuestion":
            # Always reach can_use_tool, where the questions are shown in Slack (also in auto mode).
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                           "permissionDecisionReason": "question for the user"}}
        res = policy.decide(self.cfg, tool, tin, cwd, self.memory_dir)
        if res.decision == "ask" and tool == "Bash":
            why = self.hub.allowlist.allowed(tin, cwd)
            if why:
                self.hub.audit.write(
                    session=self.rec.session_id, thread=self.rec.key, agent=inp.get("agent_id"), tool=tool,
                    input=policy.audit_summary(tool, tin), decision="allow", by="allowlist", reason=why,
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": why,
                    }
                }
            self._remember_cwd(tool_use_id or inp.get("tool_use_id"), cwd)
        if (res.decision == "ask" and self.rec.mode == "auto" and not res.force_slack
                and not self._force_slack(tool, tin, cwd)):
            # Let the CLI's auto-mode classifier decide; hard denies above still apply.
            self.hub.audit.write(
                session=self.rec.session_id, thread=self.rec.key, agent=inp.get("agent_id"), tool=tool,
                input=policy.audit_summary(tool, tin), decision="deferred", by="auto-mode-classifier",
            )
            return {}
        if res.decision != "ask":
            self.hub.audit.write(
                session=self.rec.session_id, thread=self.rec.key, agent=inp.get("agent_id"), tool=tool,
                input=policy.audit_summary(tool, tin), decision=res.decision, by="policy", reason=res.reason,
            )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": res.decision,
                "permissionDecisionReason": res.reason,
            }
        }

    def controls(self, final: bool) -> list[dict[str, Any]]:
        """Buttons under a reply. Running: stop. Finished: hand to terminal, switch mode, status."""
        key = self.rec.key
        if not final:
            out = [button(":stop_button: Stop", "sac_ctl_stop", key, "danger")]
        else:
            other = "ask" if self.rec.mode == "auto" else "auto"
            out = [button(":computer: Hand to terminal", "sac_ctl_exit", key),
                   button(f"Switch to {other} mode", f"sac_ctl_mode_{other}", key),
                   button(":bar_chart: Status", "sac_ctl_status", key)]
        if self.bg_tasks:
            out.append(button(f":gear: Stop {len(self.bg_tasks)} background task(s)", "sac_ctl_stopbg", key))
        return out

    def _remember_cwd(self, tool_use_id: str | None, cwd: Path) -> None:
        if not tool_use_id:
            return
        self._call_cwd[tool_use_id] = cwd
        while len(self._call_cwd) > 256:
            self._call_cwd.pop(next(iter(self._call_cwd)))

    def call_cwd(self, tool_use_id: str | None) -> Path:
        """Directory a Bash call runs in (the shell's, which follows `cd`), else the session's."""
        return self._call_cwd.get(tool_use_id or "") or self.cwd

    def allowlist_note(self, tool: str, tin: dict[str, Any], ctx: ToolPermissionContext) -> str:
        if tool != "Bash":
            return ""
        return self.hub.allowlist.card_note(tin, self.call_cwd(ctx.tool_use_id))

    def _learn(self, tin: dict[str, Any], tool_use_id: str | None, user: str | None) -> None:
        cwd = self.call_cwd(tool_use_id)
        self._call_cwd.pop(tool_use_id or "", None)
        ent, status = self.hub.allowlist.learn(tin, cwd, by=user or "?", session=self.rec.session_id)
        if ent is not None and status == "added":
            self.hub.audit.write(event="allowlist-add", session=self.rec.session_id, thread=self.rec.key,
                                 id=ent.id, by=user, cwd=ent.cwd, input=policy.audit_summary("Bash", tin))

    def _force_slack(self, tool: str, tin: dict[str, Any], cwd: Path | None = None) -> bool:
        """In auto mode, still require a Slack tap for commands that could leak the
        environment, send data off-node, or name protected files (after expanding
        braces, ~ and wildcards)."""
        return tool == "Bash" and bool(
            policy.bash_warnings(self.cfg, str(tin.get("command", "")), cwd or self.cwd))

    def set_model_effort(self, model: str | None = None, effort: str | None = None,
                         clear_model: bool = False, clear_effort: bool = False) -> str:
        """Change the per-thread model/effort. They are CLI start options, so a
        running CLI process is restarted before the next prompt (resuming the
        transcript). Returns when the change takes effect."""
        if model or clear_model:
            self.rec.model = None if clear_model else model
        if effort or clear_effort:
            self.rec.effort = None if clear_effort else effort
        self.hub.store.put(self.rec)
        if self.client is None:
            return "from the next message"
        self._options_stale = True
        if self.bg_tasks:
            return "once the background task(s) in this session have finished (then from the next message)"
        return "from the next message (Claude Code restarts and reloads this session's history)"

    def _transcript_file(self, sid: str | None = None) -> Path | None:
        sid = sid or self.rec.session_id
        files = list((Path.home() / ".claude" / "projects").glob(f"*/{sid}.jsonl"))
        return max(files, key=lambda q: q.stat().st_mtime) if files else None

    def last_used(self) -> tuple[str | None, str | None]:
        """(model, effort) recorded on the transcript's last assistant message."""
        f = self._transcript_file()
        model = effort = None
        if f is None:
            return None, None
        try:
            with f.open("rb") as fh:
                for line in fh:
                    if not line.startswith(b'{"') or b'"assistant"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("type") == "assistant":
                        model = e.get("requestedModel") or (e.get("message") or {}).get("model") or model
                        effort = e.get("effort") or effort
        except OSError:
            pass
        return model, effort

    async def set_mode(self, mode: str) -> None:
        self.rec.mode = mode
        self.hub.store.put(self.rec)
        if self.client is not None:
            await self.client.set_permission_mode("auto" if mode == "auto" else "default")

    async def _can_use_tool(self, tool: str, tin: dict[str, Any], ctx: ToolPermissionContext) -> PermissionResult:
        if tool == "AskUserQuestion":
            return await self.hub.ask_questions(self, tin)
        res = policy.decide(self.cfg, tool, tin, self.cwd, self.memory_dir)
        audit = dict(session=self.rec.session_id, thread=self.rec.key, agent=ctx.agent_id, tool=tool,
                     input=policy.audit_summary(tool, tin))
        if res.decision == "deny":
            self.hub.audit.write(**audit, decision="deny", by="policy", reason=res.reason)
            return PermissionResultDeny(message=res.reason)
        if res.decision == "allow":
            self.hub.audit.write(**audit, decision="allow", by="policy", reason=res.reason)
            return PermissionResultAllow(updated_input=tin)
        # "Approve all <tool>" never covers calls that must always be confirmed
        # (env dumps, off-node transfers, protected paths): review finding R2.
        if tool in self.allow_all and not self._force_slack(tool, tin, self.call_cwd(ctx.tool_use_id)):
            self.hub.audit.write(**audit, decision="allow", by="session-allow-all")
            return PermissionResultAllow(updated_input=tin)

        t0 = time.monotonic()
        if self.renderer:
            self.renderer.set_status(f"⏸ waiting for your approval: {policy.tool_line(tool, tin, self.cwd)[:80]}")
        try:
            choice, by = await self.hub.request_approval(self, tool, tin, ctx, res.reason)
        finally:
            if self.renderer:
                self.renderer.set_status("⏳ working…")
        self.hub.audit.write(**audit, decision=choice, by=by, wait_s=round(time.monotonic() - t0, 1))
        if choice in ("approve", "remember", "approve_all"):
            if choice == "approve_all":
                self.allow_all.add(tool)
            if choice == "remember" and tool == "Bash":
                self._learn(tin, ctx.tool_use_id, by)
            return PermissionResultAllow(updated_input=tin)
        if choice == "timeout":
            return PermissionResultDeny(
                message=f"No approval within {int(self.cfg.approval_timeout_s // 60)} minutes; denied automatically. "
                "Stop and wait for the user."
            )
        if choice == "cancel":
            return PermissionResultDeny(message="The user stopped this task.", interrupt=True)
        return PermissionResultDeny(message="The user denied this tool call via Slack.")

    # -- uploads -----------------------------------------------------------
    async def _offer_uploads(self, text: str) -> None:
        cands = self._turn_files + _PATH_RE.findall(text or "")
        seen: dict[Path, None] = {}
        for raw in cands:
            try:
                p = policy._resolve(raw, self.cwd)
            except Exception:
                continue
            if p in seen or p.suffix.lower() not in UPLOAD_EXTS:
                continue
            if self.hub.upload_allowed(p) is None:
                seen[p] = None
            if len(seen) >= 5:
                break
        if seen:
            await self.hub.offer_uploads(self.rec.channel, self.rec.thread_ts, list(seen), self.cwd)


def new_approval_id() -> str:
    return secrets.token_urlsafe(12)
