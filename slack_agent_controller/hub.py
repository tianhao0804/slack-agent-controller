"""Shared bot state: sessions, approvals, uploads, and Slack I/O helpers."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from claude_agent_sdk.types import PermissionResult
from slack_sdk.web.async_client import AsyncWebClient

from . import policy
from .allowlist import Allowlist
from .config import Config
from .inbox import Inbox
from .questions import SKIP_RESPONSE, PendingQuestion, new_question
from .session import Choice, PendingApproval, ThreadSession, new_approval_id
from .slackfmt import BLOCK_TEXT_LIMIT, button, escape, fit_code_block, inline_code
from .store import AuditLog, SessionRecord, SessionStore, thread_key
from .tasks import TaskStore

if TYPE_CHECKING:
    from .mirror import Mirror

log = logging.getLogger(__name__)


def _job_note(command: str) -> str:
    """For `sac-job submit` commands: what approving starts."""
    import shlex

    try:
        words = shlex.split(command)
    except ValueError:
        return ""
    for i, w in enumerate(words):
        if w.rsplit("/", 1)[-1] == "sac-job" and words[i + 1: i + 2] == ["submit"]:
            rest = words[i + 2:]
            gpus = next((rest[k + 1] for k, x in enumerate(rest[:-1]) if x == "--gpus"), None)
            gpus = gpus or next((x.split("=", 1)[1] for x in rest if x.startswith("--gpus=")), "0")
            where = f"{gpus} GPU(s), picked automatically (waits in a queue until free)" if gpus not in ("0", "") \
                else "CPU only"
            return (f":test_tube: Starts a tracked job on {escape(where)}. It runs on its own after you approve "
                    "(even if Claude or the bot restarts); the bot posts progress and the result in this thread, "
                    "and the job message has a Cancel button.")
    return ""


class Hub:
    def __init__(self, cfg: Config, web: AsyncWebClient) -> None:
        self.cfg = cfg
        self.web = web
        self.store = SessionStore(cfg.state_dir / "sessions.json")
        self.audit = AuditLog(cfg.state_dir / "audit.jsonl")
        self.allowlist = Allowlist(cfg.state_dir / "allowlist.json", cfg)
        self.tasks = TaskStore(cfg.state_dir / "tasks.json")
        self.inbox = Inbox(cfg)
        self.questions: dict[str, PendingQuestion] = {}
        self.shutting_down = False
        self.sessions: dict[str, ThreadSession] = {}
        self.approvals: dict[str, PendingApproval] = {}
        self.uploads: OrderedDict[str, tuple[Path, str, str]] = OrderedDict()
        self.fork_offers: OrderedDict[str, tuple[str, str, str, str]] = OrderedDict()
        self.turn_slots = asyncio.Semaphore(cfg.max_running_turns)  # limits replies running at once
        self.mirror: Mirror | None = None  # terminal mirror (set by the front-end when MIRROR_DIRS is set)

    # -- sessions ----------------------------------------------------------
    def get_session(self, channel: str, thread_ts: str) -> ThreadSession | None:
        key = thread_key(channel, thread_ts)
        if key in self.sessions:
            return self.sessions[key]
        rec = self.store.get(key)
        if rec is None:
            return None
        s = ThreadSession(self, rec)
        self.sessions[key] = s
        return s

    def create_session(
        self, channel: str, thread_ts: str, cwd: Path, owner: str, session_id: str | None = None
    ) -> ThreadSession:
        rec = SessionRecord(
            channel=channel,
            thread_ts=thread_ts,
            session_id=session_id or str(uuid.uuid4()),
            cwd=str(cwd),
            owner=owner,
            has_history=session_id is not None,  # attaching to an existing transcript
            mode=self.cfg.default_mode,
        )
        self.store.put(rec)
        s = ThreadSession(self, rec)
        self.sessions[rec.key] = s
        return s

    async def offer_fork(self, rec: SessionRecord, user: str, text: str, notice: str) -> None:
        if not (rec.has_history or rec.fork_from):
            await self.post(rec.channel, rec.thread_ts, notice)
            return
        token = uuid.uuid4().hex
        self.fork_offers[token] = (rec.key, rec.session_id, user, text)
        if len(self.fork_offers) > 128:
            self.fork_offers.popitem(last=False)
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": notice}},
                  {"type": "actions", "elements": [button("Send this message in a fork", "sac_fork_send", token)]}]
        if not await self.post(rec.channel, rec.thread_ts, notice, blocks=blocks):
            self.fork_offers.pop(token, None)

    def create_fork(self, channel: str, thread_ts: str, parent: SessionRecord, owner: str) -> ThreadSession:
        rec = SessionRecord(channel=channel, thread_ts=thread_ts, session_id=str(uuid.uuid4()), cwd=parent.cwd,
                            owner=owner, fork_from=parent.session_id, mode=parent.mode)
        self.store.put(rec)
        s = ThreadSession(self, rec)
        self.sessions[rec.key] = s
        return s

    async def detach(self, key: str) -> None:
        s = self.sessions.pop(key, None)
        if s:
            await s.close()
        self.store.remove(key)

    def validate_cwd(self, raw: str) -> tuple[Path | None, str]:
        """Validate a user-supplied directory; relative paths are taken from
        the first root. Returns (path, error).

        Checks use the fully resolved path, but the returned path keeps the
        caller's spelling (symlinks unresolved): Claude Code stores
        transcripts under the cwd *as given* (e.g. ~/proj, a symlink to /data/proj ->
        ~/.claude/projects/-home-user-proj), so resuming with the resolved
        /data/proj path would not find the session.
        """
        raw = raw.strip()
        base = self.cfg.cwd_roots[0]
        p = Path(os.path.expanduser(raw))
        given = Path(os.path.abspath(p if p.is_absolute() else base / p))
        real = Path(os.path.realpath(given))
        if not real.is_dir():
            return None, f"{given} is not an existing directory"
        if not self.cfg.under_roots(real):
            roots = ", ".join(str(r) for r in self.cfg.cwd_roots)
            return None, f"{real} is outside the allowed roots ({roots})"
        if policy.is_sensitive(self.cfg, real) or policy.is_sensitive(self.cfg, given):
            return None, f"{given} is a protected path"
        return given, ""

    @staticmethod
    def bedrock_model_ok(model: str) -> bool:
        m = model.lower()
        return m.startswith(("us.", "eu.", "apac.", "global.", "anthropic.", "arn:aws:")) or "bedrock" in m

    async def shutdown(self) -> None:
        self.shutting_down = True  # unfinished messages stay recorded and are reported after the restart
        await asyncio.gather(*(s.close() for s in self.sessions.values()), return_exceptions=True)

    async def report_unfinished(self) -> int:
        """After a (re)start: tell each thread which of its messages did not complete.
        Nothing is resent; an interrupted reply may already have done part of the work."""
        recs = self.tasks.take_all()
        by_thread: dict[tuple[str, str], list[Any]] = {}
        for r in recs:
            by_thread.setdefault((r.channel, r.thread_ts), []).append(r)
        for (channel, thread_ts), items in by_thread.items():
            users = sorted({r.user for r in items if r.user})
            lines = [" ".join(f"<@{u}>" for u in users) + " :warning: The bot restarted, so these messages did not "
                     "complete:"]
            for r in items:
                quote = "\n".join("> " + ln for ln in escape(r.text[:300]).splitlines()[:4]) or "> (empty)"
                if r.status == "queued":
                    lines.append("*Not started* (it was waiting in the queue); send it again if you still want it:\n"
                                 + quote)
                else:
                    lines.append("*Interrupted while running*: the reply was cut off and Claude may already have done "
                                 "part of it (commands, file edits). Check before sending it again:\n" + quote)
            await self.post(channel, thread_ts, "\n".join(lines)[:3900])
            self.audit.write(event="restart-report", thread=f"{channel}:{thread_ts}",
                             tasks=[{"status": r.status, "session": r.session_id} for r in items])
        return len(recs)

    # -- Slack helpers -----------------------------------------------------
    async def post(self, channel: str, thread_ts: str | None, text: str, **kw: Any) -> str | None:
        try:
            resp = await self.web.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text, **kw)
            return resp["ts"]
        except Exception:
            log.exception("chat.postMessage failed")
            return None

    async def upload_content(self, channel: str, thread_ts: str, content: str, filename: str, title: str) -> None:
        try:
            await self.web.files_upload_v2(
                channel=channel, thread_ts=thread_ts, content=content, filename=filename, title=title
            )
        except Exception:
            log.exception("files_upload_v2 (content) failed")

    def upload_allowed(self, p: Path) -> str | None:
        """Returns an error string, or None if the file may be uploaded."""
        if not p.is_file():
            return f"{p} is not a file"
        if not self.cfg.under_roots(p) or policy.is_sensitive(self.cfg, p):
            return f"{p} is outside the allowed roots or protected"
        mb = p.stat().st_size / 1e6
        if mb > self.cfg.upload_max_mb:
            return f"{p} is {mb:.1f} MB (limit {self.cfg.upload_max_mb:.0f} MB)"
        return None

    async def upload_file(self, channel: str, thread_ts: str, p: Path, user: str) -> str | None:
        err = self.upload_allowed(p)
        self.audit.write(event="upload", path=str(p), by=user, decision="deny" if err else "allow", reason=err)
        if err:
            return err
        try:
            await self.web.files_upload_v2(
                channel=channel, thread_ts=thread_ts, file=str(p), filename=p.name, title=p.name
            )
        except Exception as e:
            log.exception("upload failed")
            return f"upload failed: {e}"
        return None

    async def offer_uploads(self, channel: str, thread_ts: str, paths: list[Path], cwd: Path) -> None:
        buttons: list[dict[str, Any]] = []
        for p in paths:
            tok = new_approval_id()
            self.uploads[tok] = (p, channel, thread_ts)
            label = str(p.relative_to(cwd)) if p.is_relative_to(cwd) else p.name
            buttons.append({
                "type": "button", "action_id": f"sac_upload_{len(buttons)}", "value": tok,
                "text": {"type": "plain_text", "text": f"📎 {label}"[:75]},
            })
        while len(self.uploads) > 500:
            self.uploads.popitem(last=False)
        await self.post(
            channel, thread_ts, "Files mentioned in this reply:",
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": "Attach files mentioned in this reply?"}},
                {"type": "actions", "elements": buttons},
            ],
        )

    # -- approvals ---------------------------------------------------------
    async def request_approval(
        self, sess: ThreadSession, tool: str, tin: dict[str, Any], ctx: ToolPermissionContext, reason: str
    ) -> tuple[Choice, str | None]:
        aid = new_approval_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[Choice, str | None]] = loop.create_future()
        rec = sess.rec
        owner = sess.turn_owner or rec.owner
        line = policy.redact_log(policy.tool_line(tool, tin, sess.cwd))
        detail = policy.redact_log(policy.approval_detail(tool, tin, sess.cwd))
        full = policy.redact_log(policy.approval_full(tool, tin, sess.cwd))
        detail_block, cut = fit_code_block(detail)
        warns = (policy.bash_warnings(self.cfg, str(tin.get("command", "")), sess.call_cwd(ctx.tool_use_id))
                 if tool == "Bash" else [])
        note = sess.allowlist_note(tool, tin, ctx)
        title = ctx.title or f"Claude wants to use {tool}"
        timeout_min = int(self.cfg.approval_timeout_s // 60)

        header = f"<@{owner}> :lock: *{escape(title)}*"
        if ctx.agent_id:
            header += f"  _(sub-agent {escape(ctx.agent_id[:8])})_"
        if ctx.blocked_path:
            header += f"\nBlocked path: {inline_code(ctx.blocked_path)}"
        if ctx.decision_reason or reason:
            header += f"\n_{escape(ctx.decision_reason or reason)}_"
        blocks: list[dict[str, Any]] = [
            {"type": "section", "text": {"type": "mrkdwn", "text": header[:BLOCK_TEXT_LIMIT]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": detail_block}},
        ]
        job_note = _job_note(str(tin.get("command", ""))) if tool == "Bash" else ""
        if job_note:
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": job_note}]})
        if warns:
            shown = [f":warning: *{escape(w[:200])}*" for w in warns[:5]]
            if len(warns) > 5:
                shown.append(f"_…and {len(warns) - 5} more_")
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(shown)[:BLOCK_TEXT_LIMIT]}})
        cwd = sess.call_cwd(ctx.tool_use_id) if tool == "Bash" else sess.cwd
        can_remember = tool == "Bash" and self.allowlist.can_remember(tin, cwd)
        actions = [button("Run once", "sac_approve", aid, "primary")]
        if can_remember:
            actions.append(button("Run & remember here", "sac_remember", aid))
        actions.append(button("Deny", "sac_deny", aid, "danger"))
        blocks += [
            {"type": "context", "elements": [{"type": "mrkdwn", "text":
                f"session `{rec.session_id[:8]}` · cwd `{escape(str(cwd))}` · auto-deny in {timeout_min} min"
                + (f"\n{note}" if note else "")}]},
            {"type": "actions", "block_id": f"sac_appr_{aid}", "elements": [
                *actions,
                {"type": "overflow", "action_id": "sac_approval_more", "options": [
                    {"text": {"type": "plain_text", "text": f"Allow all {tool} this session"[:75]}, "value": aid}],
                 "confirm": {
                     "title": {"type": "plain_text", "text": "Approve all?"},
                     "text": {"type": "plain_text",
                              "text": f"Every {tool} call in this session will run without asking, except ones flagged with a warning (they always ask)."[:300]},
                     "confirm": {"type": "plain_text", "text": "Approve all"},
                     "deny": {"type": "plain_text", "text": "Cancel"},
                 }},
            ]},
        ]
        pending = PendingApproval(id=aid, session=sess, tool=tool, future=fut, channel=rec.channel, line=line,
                                  can_remember=can_remember)
        self.approvals[aid] = pending
        sess.pending.add(aid)
        # The @mention must be in the top-level text too: that is what phone notifications show.
        pending.ts = await self.post(rec.channel, rec.thread_ts, f"<@{owner}> Approval needed: {line}"[:300],
                                     blocks=blocks)
        if pending.ts is None:  # could not even ask -> deny
            self.approvals.pop(aid, None)
            sess.pending.discard(aid)
            return "deny", "system(post-failed)"
        if cut or full != detail:  # the card shows a preview: attach everything that will run
            await self.upload_content(rec.channel, rec.thread_ts, full, f"{tool.lower()}-request.txt",
                                      f"Full {tool} request")
        t0 = time.monotonic()
        try:
            choice, by = await asyncio.wait_for(asyncio.shield(fut), timeout=self.cfg.approval_timeout_s)
        except TimeoutError:
            choice, by = "timeout", None
        finally:
            self.approvals.pop(aid, None)
            sess.pending.discard(aid)

        outcome = {
            "approve": ":white_check_mark: Approved once",
            "remember": ":white_check_mark: Approved with permission to remember this command here",
            "approve_all": f":white_check_mark: Approved all {tool} for this session",
            "deny": ":no_entry: Denied",
            "timeout": f":hourglass: No response in {timeout_min} min — denied",
            "cancel": ":stop_button: Cancelled (task stopped)",
        }[choice]
        who = f" by <@{by}>" if by and by != "system" else ""
        final_blocks = blocks[:-1] + [{"type": "context", "elements": [{"type": "mrkdwn",
                                       "text": f"{outcome}{who} · {int(time.monotonic() - t0)}s"}]}]
        final_blocks[0] = {"type": "section", "text": {"type": "mrkdwn",
                           "text": f":lock: *{escape(title)}*"[:BLOCK_TEXT_LIMIT]}}  # drop the @mention
        try:
            await self.web.chat_update(channel=rec.channel, ts=pending.ts, text=f"{outcome}: {line}"[:300],
                                       blocks=final_blocks)
        except Exception:
            log.exception("failed to update approval message")
        return choice, by

    def resolve_approval(self, aid: str, choice: Choice, user: str, check_user: bool = True) -> str:
        """Resolve a pending approval. Returns a short status string."""
        if check_user and user not in self.cfg.allowed_users:
            log.warning("approval click from non-allowlisted user %s ignored", user)
            self.audit.write(event="approval-click-rejected", user=user, approval=aid)
            return "forbidden"
        p = self.approvals.get(aid)
        if p is None and choice == "cancel" and (q := self.questions.get(aid)) is not None:
            if not q.future.done():
                q.future.set_result(("cancel", None))  # !stop also withdraws an open question
            return "ok"
        if p is None or p.future.done():
            return "expired"
        if choice == "remember" and not p.can_remember:
            return "forbidden"
        p.future.set_result((choice, user))
        return "ok"

    # -- clarifying questions (AskUserQuestion) ------------------------------
    def question_for(self, session_key: str) -> PendingQuestion | None:
        return next((q for q in self.questions.values() if q.session_key == session_key and not q.future.done()), None)

    async def ask_questions(self, sess: ThreadSession, tin: dict[str, Any]) -> PermissionResult:
        """Show Claude's questions as a card and wait for the answers (or Skip / timeout / !stop)."""
        rec = sess.rec
        owner = sess.turn_owner or rec.owner
        q = new_question(rec.key, owner, tin.get("questions"), asyncio.get_running_loop())
        if not q.questions:
            return PermissionResultDeny(message="AskUserQuestion needs at least one question.")
        q.channel = rec.channel
        self.questions[q.id] = q
        sess.pending.add(q.id)
        timeout_min = int(self.cfg.approval_timeout_s // 60)
        first = policy.redact_log(q.text_of(0))[:200]
        q.ts = await self.post(rec.channel, rec.thread_ts, f"<@{owner}> Claude asks: {first}", blocks=q.blocks())
        if q.ts is None:
            self.questions.pop(q.id, None)
            sess.pending.discard(q.id)
            return PermissionResultDeny(message="Could not show the question in Slack; ask in plain text instead.")
        if sess.renderer:
            sess.renderer.set_status("❓ waiting for your answer")
        try:
            kind, value = await asyncio.wait_for(asyncio.shield(q.future), timeout=self.cfg.approval_timeout_s)
        except TimeoutError:
            kind, value = "timeout", None
        finally:
            self.questions.pop(q.id, None)
            sess.pending.discard(q.id)
            if sess.renderer:
                sess.renderer.set_status("⏳ working…")
        final = {"answers": "answered", "skip": "skipped", "cancel": "withdrawn (task stopped)",
                 "timeout": f"no answer in {timeout_min} min"}[kind]
        try:
            await self.web.chat_update(channel=rec.channel, ts=q.ts, text=f"Claude asked: {first} ({final})",
                                       blocks=q.blocks(final=final))
        except Exception:
            log.exception("failed to update question card")
        self.audit.write(event="question", session=rec.session_id, thread=rec.key, outcome=kind,
                         questions=len(q.questions))
        if kind == "answers":
            return PermissionResultAllow(updated_input={"questions": q.questions, "answers": value})
        if kind == "skip":
            return PermissionResultAllow(updated_input={"questions": q.questions, "answers": {},
                                                        "response": SKIP_RESPONSE})
        if kind == "cancel":
            return PermissionResultDeny(message="The user stopped this task.", interrupt=True)
        return PermissionResultDeny(message=f"No answer within {timeout_min} minutes. Stop and wait for the user.")

    def answer_question(self, qid: str, user: str, action: dict[str, Any]) -> tuple[PendingQuestion | None, str]:
        """Apply a click on a question card. Returns (question, status)."""
        if user not in self.cfg.allowed_users:
            log.warning("question click from non-allowlisted user %s ignored", user)
            self.audit.write(event="question-click-rejected", user=user, question=qid)
            return None, "forbidden"
        q = self.questions.get(qid)
        if q is None or q.future.done():
            return None, "expired"
        aid = str(action.get("action_id", ""))
        if aid == "sac_qa_skip":
            q.future.set_result(("skip", None))
            return q, "ok"
        parts = str(action.get("value", "")).split("|")
        if aid.startswith("sac_qa_multi_"):
            qi = int(aid.rsplit("_", 1)[1])
            picked = []
            labels = q.labels(qi)
            for o in action.get("selected_options") or []:
                idx = int(str(o.get("value", "")).rsplit("|", 1)[-1])
                if idx < len(labels):
                    picked.append(labels[idx])
            q.selections[qi] = picked
            return q, "selected"
        if aid.startswith("sac_qa_submit_"):
            qi = int(parts[1])
            if not q.selections.get(qi):
                return q, "nothing-selected"
            q.answer(qi, list(q.selections[qi]))
            return q, "ok"
        if aid.startswith("sac_qa_pick_") and len(parts) == 3:
            qi, opt = int(parts[1]), int(parts[2])
            labels = q.labels(qi)
            if opt < len(labels):
                q.answer(qi, labels[opt])
                return q, "ok"
        return q, "ignored"
    # -- status ------------------------------------------------------------
    async def _run(self, *cmd: str, timeout: float = 15.0) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return out.decode(errors="replace").strip()
        except FileNotFoundError:
            return f"{cmd[0]}: not found"
        except TimeoutError:
            proc.kill()
            return f"{cmd[0]}: timed out"

    async def status_text(self, sess: ThreadSession | None) -> str:
        gpu = await self._run(
            "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        )
        rows = []
        for line in gpu.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) == 6:
                rows.append(f"{f[0]:>2} {f[1][:16]:<16} {f[2]:>3}%  {int(f[3]) / 1024:5.1f}/{int(f[4]) / 1024:4.0f}G  {f[5]}°C")
            else:
                rows.append(line)
        parts = ["*GPUs*", "```" + escape("\n".join(rows) or gpu) + "```"]
        if self.cfg.squeue_path:
            sq = await self._run(self.cfg.squeue_path, "-u", os.environ.get("USER", ""), "-o",
                                 "%.10i %.18j %.8T %.10M %.4D %R")
            parts += ["*squeue*", "```" + escape(sq[:2500]) + "```"]
        else:
            parts.append("_squeue: not available on this node (set SQUEUE_PATH)_")
        if sess:
            r = sess.rec
            state = "running" if sess.busy else ("connected (idle)" if sess.connected else "not connected")
            state += f" · mode {r.mode} · model {r.model or self.cfg.model or 'default'}"
            state += f" · effort {r.effort or self.cfg.default_effort or 'default'}"
            if self.mirror and (lk := self.mirror.locked(r.session_id)):
                state += f" · :lock: open in a terminal (pid {lk.pid or '?'}, since {time.strftime('%H:%M', time.localtime(lk.opened_at))})"
            parts += [
                "*This session*",
                f"id `{r.session_id}` · {state}\ncwd `{escape(r.cwd)}`\n"
                f"turns {r.turns} · tokens in {r.input_tokens:,} / out {r.output_tokens:,} · "
                f"est. cost ${r.cost_usd:.2f}"
                + (f"\nauto-approved tools: {', '.join(sorted(sess.allow_all))}" if sess.allow_all else "")
                + ("\nbackground tasks: " + "; ".join(escape(d[:60]) for d in sess.bg_tasks.values())
                   if sess.bg_tasks else ""),
                f"_terminal: `cd {escape(r.cwd)} && claude --resume {r.session_id}`_",
            ]
        running = sum(1 for s in self.sessions.values() if s.busy)
        parts.append(f"_bot: {len(self.store.records)} sessions stored, {running} running_")
        if self.cfg.token_expires:
            days = (self.cfg.token_expires - dt.date.today()).days
            parts.append(f"_API key ({self.cfg.provider}) expires {self.cfg.token_expires} ({days} days)_")
        return "\n".join(parts)

    async def check_stalls(self) -> int:
        """@mention the owner once when a running reply has shown no activity for
        STALL_NOTIFY_MIN (not counting time spent waiting for an approval)."""
        now = time.monotonic()
        sent = 0
        for s in list(self.sessions.values()):
            if not s.busy or s.pending or s.stall_notified or now - s.last_event < self.cfg.stall_notify_s:
                continue
            s.stall_notified = True
            mins = int((now - s.last_event) // 60)
            running = list(s.open_tools.values())
            if running:
                detail = (f"Claude is still running {inline_code(running[-1][:120])}"
                          + (f" (+{len(running) - 1} more)" if len(running) > 1 else "")
                          + "; a long command can be normal.")
            else:
                detail = "No tool is running, so it may be stuck waiting on the model or the CLI."
            await self.post(s.rec.channel, s.rec.thread_ts,
                            f"<@{s.turn_owner or s.rec.owner}> :hourglass: No activity in this reply for {mins} min. "
                            f"{detail} `!stop` to stop it.")
            self.audit.write(event="stall-notice", session=s.rec.session_id, thread=s.rec.key, minutes=mins,
                             open_tools=len(running))
            sent += 1
        return sent

    async def stall_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self.check_stalls()
            except Exception:
                log.exception("stall check failed")

    async def token_expiry_loop(self) -> None:
        """Daily DM reminder when the provider API key is about to expire (API_KEY_EXPIRES)."""
        warned: dt.date | None = None
        while True:
            exp = self.cfg.token_expires
            if exp:
                days = (exp - dt.date.today()).days
                if days <= 3 and warned != dt.date.today():
                    warned = dt.date.today()
                    for u in self.cfg.allowed_users:
                        await self.post(u, None, f":key: The {self.cfg.provider} API key expires on {exp} ({days} days). Update "
                                        f"it and API_KEY_EXPIRES in `{self.cfg.env_file}`,"
                                        " then `systemctl --user restart slack-agent-controller`.")
            await asyncio.sleep(6 * 3600)
