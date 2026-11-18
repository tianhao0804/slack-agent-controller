"""Slack Bolt (Socket Mode) front-end: events, commands, button actions."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import signal
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from claude_agent_sdk import get_session_info, list_sessions
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.web.async_client import AsyncWebClient

from . import home, importer, policy
from .config import EFFORTS, Config
from .hub import Hub
from .inbox import cleanup_loop, prompt_with_files
from .jobs import JobWatcher
from .questions import PendingQuestion
from .session import SAC_JOB, STOP_WAIT_S, ThreadSession
from .slackfmt import button, escape, inline_code, slack_to_plain
from .store import thread_key

log = logging.getLogger(__name__)

HELP = """\
*Agent Controller* (Claude, via the Agent SDK) — message me to start a session; reply in the thread to continue it.
• `!new <dir> [prompt]` — new session in `<dir>` (relative paths are under {root})
• `!stop` — stop the running reply and drop queued messages · `!stop bg` — stop background tasks · `!stop all` — both
• `!status` — GPUs, squeue, this session's id / tokens / cost
• `!sessions` — sessions with titles, state and Open buttons · `!sessions all` — also recent terminal sessions
• `!jobs` — long computations started with `sac-job` (running first), with their logs
• `!projects` — your project shortcuts (PROJECTS); `!new <alias> [message]` starts a session there
• `!resume <session_id>` — attach a terminal Claude Code session to a new thread
• `!import <id-prefix> [...]` — import sessions as threads here, with their last 5 exchanges
• `!fork` — (in a thread) branch this session into a new thread; the original stays untouched
• `!upload <path>` — attach a file from the session's directory
• `!mode [auto|ask]` — per-thread permissions: `ask` = approve with buttons, `auto` = Claude Code auto-mode classifier
• `!allowlist [remove <id>|clear]` — Bash commands you chose to remember (exact text + directory) and no longer ask; list or revoke them
• `!end` — (in a thread) end this session: unbind the thread and stop mirroring it (transcript kept)
• `!exit` — (in a thread) close this thread's Claude Code process now so the session can be used in a terminal; reply here later to continue
• `!unlock` — (in a thread) release a stale terminal lock (refused while the terminal process still runs; `!unlock force` overrides)
• `!model [opus|sonnet|haiku|<model id>|default]` · `!effort [low|medium|high|xhigh|max|default]` — per-thread model / effort; without an argument: show current and last used
• `!help` — this message
`!end` in a thread (or deleting a root message you wrote) ends that session; the transcript is kept for `!resume`.
Tool calls with side effects ask for approval with buttons (auto-deny after {timeout} min)."""

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


class SlackFrontend:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.app = AsyncApp(token=cfg.slack_bot_token, logger=logging.getLogger("slack_bolt"))
        web: AsyncWebClient = self.app.client
        web.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=3))
        self.hub = Hub(cfg, web)
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.jobs: JobWatcher | None = None  # started in run()
        self._register()

    # -- plumbing ----------------------------------------------------------
    def _spawn(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _dedupe(self, key: str) -> bool:
        if key in self._seen:
            return True
        self._seen[key] = None
        while len(self._seen) > 2000:
            self._seen.popitem(last=False)
        return False

    def _allowed_place(self, event: dict[str, Any]) -> bool:
        return event.get("channel_type") == "im" or event.get("channel") in self.cfg.allowed_channels

    def _register(self) -> None:
        app = self.app

        @app.event("message")
        async def on_message(event: dict[str, Any]) -> None:
            await self._on_message(event)

        @app.action(re.compile(r"^sac_(approve|remember|deny|approve_all|approval_more)$"))
        async def on_approval(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            user = body.get("user", {}).get("id", "")
            choice = action["action_id"].removeprefix("sac_")
            aid = action.get("value", "")
            if choice == "approval_more":
                choice = "approve_all"
                aid = (action.get("selected_option") or {}).get("value", "")
            status = self.hub.resolve_approval(aid, choice, user)  # type: ignore[arg-type]
            if status == "expired":
                ch = body.get("channel", {}).get("id")
                ts = body.get("message", {}).get("ts")
                if ch and ts and user in self.cfg.allowed_users:
                    try:
                        await self.hub.web.chat_update(
                            channel=ch, ts=ts, text="Approval expired",
                            blocks=[{"type": "context", "elements": [{"type": "mrkdwn",
                                     "text": ":hourglass: This approval request is no longer pending."}]}],
                        )
                    except Exception:
                        log.exception("failed to mark approval expired")

        @app.action(re.compile(r"^sac_ctl_(stop|stopkeep|stopclear|stopbg|exit|mode(?:_ask|_auto)?|status)$"))
        async def on_control(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            await self._on_control(body, action)

        @app.action(re.compile(r"^sac_q_(now|keep|withdraw)$"))
        async def on_queue_choice(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            await self._on_queue_choice(body, action)

        @app.action("sac_fork_send")
        async def on_fork_send(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            await self._on_fork_send(body, action)

        @app.action(re.compile(r"^sac_proj_new_[a-z0-9_-]+$"))
        async def on_project_new(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            await self._on_project_new(body, action)

        @app.action(re.compile(r"^sac_open_"))
        async def on_open(ack: Any) -> None:
            await ack()  # URL buttons open the thread in Slack; nothing to do here

        @app.action(re.compile(r"^sac_qa_"))
        async def on_question(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            await self._on_question(body, action)

        @app.event("app_home_opened")
        async def on_home_opened(event: dict[str, Any]) -> None:
            if event.get("tab") == "home":
                await self._publish_home(str(event.get("user", "")))

        @app.action(re.compile(r"^sac_home_(refresh|new|proj_[a-z0-9_-]+)$"))
        async def on_home_action(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            await self._on_home_action(body, action)

        @app.action(re.compile(r"^sac_home_open_"))
        async def on_home_open(ack: Any) -> None:
            await ack()  # URL buttons: Slack opens the link

        @app.action(re.compile(r"^sac_job_(log|cancel|full|analyze)(_h[0-9a-f]{6})?$"))
        async def on_job(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            if self.jobs is not None:
                await self.jobs.on_action(body, action)

        @app.action(re.compile(r"^sac_upload_\d+$"))
        async def on_upload(ack: Any, body: dict[str, Any], action: dict[str, Any]) -> None:
            await ack()
            user = body.get("user", {}).get("id", "")
            if user not in self.cfg.allowed_users:
                log.warning("upload click from non-allowlisted user %s ignored", user)
                return
            item = self.hub.uploads.get(action.get("value", ""))
            if not item:
                return
            path, channel, thread_ts = item
            err = await self.hub.upload_file(channel, thread_ts, path, user)
            if err:
                await self.hub.post(channel, thread_ts, f":warning: {escape(err)}")

    async def _publish_home(self, user: str) -> None:
        if not user:
            return
        try:
            if user not in self.cfg.allowed_users:
                log.info("Home tab opened by non-allowlisted user %s", user)
                view = home.denied_view()
            else:
                view = await home.build_home(self, user)
            await self.hub.web.views_publish(user_id=user, view=view)
        except Exception:
            log.exception("failed to publish the Home tab")

    async def _dm_channel(self, user: str) -> str | None:
        try:
            resp = await self.hub.web.conversations_open(users=user)
            return str(resp["channel"]["id"])
        except Exception:
            log.exception("could not open a DM with %s", user)
            return None

    async def _on_home_action(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        user = body.get("user", {}).get("id", "")
        if user not in self.cfg.allowed_users:
            log.warning("Home click from non-allowlisted user %s ignored", user)
            return
        aid = str(action.get("action_id", ""))
        if aid == "sac_home_new" or aid.startswith("sac_home_proj_"):
            is_proj = aid.startswith("sac_home_proj_")
            d = dict(self.cfg.projects).get(str(action.get("value", ""))) if is_proj else None
            if is_proj and not d:
                return
            dm = await self._dm_channel(user)
            if dm:
                root = await self.hub.post(dm, None, ":sparkles: Starting a new session…")
                if root:
                    await self._cmd_new(dm, root, None, user, d or "")
        await self._publish_home(user)

    async def _on_question(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        user = body.get("user", {}).get("id", "")
        value = str(action.get("value") or "")
        if not value and action.get("selected_options") is not None:
            opts = action.get("selected_options") or []
            value = str(opts[0].get("value", "")) if opts else ""
        qid = value.split("|", 1)[0]
        if action.get("action_id", "").startswith("sac_qa_attach_"):
            question = self.hub.questions.get(qid)
            parts = value.split("|")
            if user not in self.cfg.allowed_users or len(parts) != 3 or not parts[2].isdigit():
                return
            channel = body.get("channel", {}).get("id")
            thread = body.get("message", {}).get("thread_ts")
            if not channel or not thread:
                return
            reply = question.attached_replies.get(parts[1]) if question else None
            if reply and reply[0] != user:
                return
            if (question and reply and question.session_key == thread_key(channel, thread)
                    and question.answer(int(parts[2]), reply[1], label=reply[2])):
                del question.attached_replies[parts[1]]
                await self._refresh_answer(question, channel, thread)
            else:
                await self.hub.post(channel, thread, "That answer slot is closed. Select another open question or resend the attachment.")
            return
        if not qid and action.get("action_id", "").startswith("sac_qa_multi_"):
            # every option unticked: find the question by the card's message ts
            ts = body.get("message", {}).get("ts")
            qid = next((q.id for q in self.hub.questions.values() if q.ts == ts), "")
        q, status = self.hub.answer_question(qid, user, action)
        if status == "expired":
            ch, ts = body.get("channel", {}).get("id"), body.get("message", {}).get("ts")
            if ch and ts and user in self.cfg.allowed_users:
                try:
                    await self.hub.web.chat_update(channel=ch, ts=ts, text="Question closed", blocks=[
                        {"type": "context", "elements": [{"type": "mrkdwn",
                         "text": ":hourglass: This question is no longer open."}]}])
                except Exception:
                    log.exception("failed to mark question closed")
            return
        if q is not None and status in ("ok", "nothing-selected") and q.ts and not q.future.done():
            try:  # show progress on a multi-question card
                await self.hub.web.chat_update(channel=q.channel, ts=q.ts, text="Claude asks", blocks=q.blocks())
            except Exception:
                log.exception("failed to update question card")

    async def _on_queue_choice(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        """Buttons on a "Queued" notice: run this message now (interrupting) or keep it queued."""
        user = body.get("user", {}).get("id", "")
        if user not in self.cfg.allowed_users:
            log.warning("queue click from non-allowlisted user %s ignored", user)
            return
        key, _, tid = str(action.get("value", "")).partition("|")
        channel = body.get("channel", {}).get("id") or key.partition(":")[0]
        msg_ts = body.get("message", {}).get("ts")
        sess = self.hub.sessions.get(key)
        rec = self.hub.tasks.records.get(tid)

        async def say(text: str, keep_withdraw: bool = False) -> None:
            blocks: list[dict[str, Any]] = [{"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}]
            if keep_withdraw:
                blocks.append({"type": "actions", "elements": [
                    button("Withdraw this message", "sac_q_withdraw", f"{key}|{tid}")]})
            if msg_ts:
                try:
                    await self.hub.web.chat_update(channel=channel, ts=msg_ts, text=text, blocks=blocks)
                    return
                except Exception:
                    log.exception("failed to update queue notice")
            await self.hub.post(channel, key.partition(":")[2], text, blocks=blocks)

        if sess is None or rec is None or tid not in sess.queued_task_ids():
            await say("This message is no longer waiting (it already started or was dropped).")
            return
        if action.get("action_id") == "sac_q_keep":
            await say(f":ballot_box_with_check: Kept in the queue (by <@{user}>). {inline_code(policy.redact_log(rec.text)[:120])}",
                      keep_withdraw=True)
            return
        if action.get("action_id") == "sac_q_withdraw":
            if sess.drop_queued(tid):
                await say("Withdrawn. This message will not run.")
            else:
                await say("This message already left the queue.")
            return
        if not sess.promote(tid):
            await say("This message already left the queue.")
            return
        r = await sess.stop(user, clear_queue=False)
        self.hub.audit.write(event="queue-interrupt", thread=key, by=user, stopped=r.stopped, error=r.error)
        if r.was_running and not r.stopped:
            await say(":warning: Asked the running reply to stop, but it has not ended yet; your message is next "
                      "in line.")
        else:
            await say(f":fast_forward: Interrupted the running reply (by <@{user}>); this message runs now.")

    async def _on_control(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        """A button under a reply: same as typing the command in that thread."""
        user = body.get("user", {}).get("id", "")
        if user not in self.cfg.allowed_users:
            log.warning("control click from non-allowlisted user %s ignored", user)
            self.hub.audit.write(event="control-click-rejected", user=user, action=action.get("action_id"))
            return
        key, _, scope = str(action.get("value", "")).partition("|")
        channel, _, thread_ts = key.partition(":")
        rec = self.hub.store.get(key)
        if rec is None:
            if channel and thread_ts:
                await self.hub.post(channel, thread_ts, "This thread is no longer bound to a session.")
            return
        what = str(action.get("action_id", "")).removeprefix("sac_ctl_")
        ts = str(body.get("message", {}).get("ts") or thread_ts)
        self.hub.audit.write(event="control-click", action=what, thread=key, by=user)
        if what in ("stop", "stopkeep", "stopclear"):
            origin, _, snapshot = scope.partition("|")
            sess = self.hub.sessions.get(key)
            if sess and sess.renderer and sess.renderer.ts and sess.renderer.ts != (origin or ts):
                await self.hub.post(channel, thread_ts, "This stop button belongs to an earlier reply.")
                return
            waiting = sess.queued_task_ids() if sess else []
            count = len(waiting)
            signature = hashlib.sha256(",".join(waiting).encode()).hexdigest()[:16]
            if what != "stop" and (not sess or not sess.busy):
                await self.hub.post(channel, thread_ts, "That reply has already ended.")
                return
            if what == "stopclear" and snapshot != signature:
                await self.hub.post(channel, thread_ts, "Queue changed. Tap Stop on the running reply to review it again.")
                return
            if what == "stop" and count:
                value = f"{key}|{ts}|{signature}"
                label = f"{count} queued message(s)"
                await self.hub.post(channel, thread_ts, f"Stop this reply? {label}.", blocks=[
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Stop this reply? {label}."}},
                    {"type": "actions", "elements": [
                        button(f"Stop reply, keep {count}", "sac_ctl_stopkeep", value),
                        button(f"Stop reply, clear {count}", "sac_ctl_stopclear", value, "danger")]}])
                return
            await self._cmd_stop(channel, ts, thread_ts, user, "" if what == "stopclear" else "keep",
                                 from_button=True)
        elif what == "stopbg":
            await self._cmd_stop(channel, ts, thread_ts, user, "bg")
        elif what == "exit":
            await self._cmd_exit(channel, ts, thread_ts, user, "")
        elif what in ("mode_ask", "mode_auto", "mode"):
            target = what.removeprefix("mode_")
            if what == "mode":  # cards posted by older versions only carried the label
                label = (action.get("text") or {}).get("text", "")
                target = {"Switch to ask mode": "ask", "Switch to auto mode": "auto"}.get(label, "")
            await self._cmd_mode(channel, ts, thread_ts, user, target)
        elif what == "status":
            await self._cmd_status(channel, ts, thread_ts, user, "")

    # -- message handling --------------------------------------------------
    async def _on_message(self, event: dict[str, Any]) -> None:
        subtype = event.get("subtype")
        if subtype in ("message_deleted", "message_changed"):
            await self._on_deleted(event)
            return
        if subtype not in (None, "file_share", "thread_broadcast") or event.get("bot_id"):
            return
        user = event.get("user", "")
        channel = event.get("channel", "")
        if user not in self.cfg.allowed_users:
            log.warning("ignored message from non-allowlisted user %s in %s", user, channel)
            self.hub.audit.write(event="message-rejected", user=user, channel=channel)
            return
        if not self._allowed_place(event):
            log.info("ignored message in non-allowed channel %s", channel)
            return
        ts = event["ts"]
        if self._dedupe(event.get("client_msg_id") or f"{channel}:{ts}"):
            return
        text = slack_to_plain(event.get("text", "")).strip()
        files = [f for f in event.get("files") or [] if isinstance(f, dict)]
        if not text and not files:
            return
        thread_ts = event.get("thread_ts")
        self._spawn(self._dispatch(channel, ts, thread_ts, user, text, files))

    async def _on_deleted(self, event: dict[str, Any]) -> None:
        """Deleting a session's thread-root message ends that session.

        Slack sends ``message_deleted`` for a root without replies, and
        ``message_changed`` with ``message.subtype == "tombstone"`` for a root
        that has replies (the thread survives with a placeholder). Deletion
        events do not say who deleted the message, so the original author
        must be allowlisted (in a DM only the author can delete it).
        """
        channel = event.get("channel", "")
        prev = event.get("previous_message") or {}
        if event.get("subtype") == "message_deleted":
            ts = event.get("deleted_ts", "")
        else:
            msg = event.get("message") or {}
            if msg.get("subtype") != "tombstone":
                return  # an ordinary edit
            ts = msg.get("ts", "")
        if not ts or not self._allowed_place(event):
            return
        key = thread_key(channel, ts)
        rec = self.hub.store.get(key)
        if rec is None:
            return  # not a session root (e.g. a reply inside a thread)
        author = prev.get("user", "")
        if author not in self.cfg.allowed_users:
            log.warning("ignored deletion of session root %s authored by %s", key, author)
            return
        if self._dedupe(f"deleted:{key}"):
            return
        self._spawn(self._end_session(rec.channel, rec.thread_ts, rec.session_id, rec.cwd, author))

    async def _end_session(self, channel: str, thread_ts: str, session_id: str, cwd: str, user: str,
                           reason: str = "thread root deleted") -> None:
        key = thread_key(channel, thread_ts)
        sess = self.hub.sessions.get(key)
        was_running = bool(sess and sess.busy)
        await self.hub.detach(key)  # interrupts, denies pending approvals, stops the CLI, drops mapping
        if self.hub.mirror:
            self.hub.mirror.forget_thread(session_id)
        log.info("session %s ended (%s) by %s", session_id, reason, user)
        self.hub.audit.write(event="session-ended", session=session_id, thread=key, by=user,
                             reason=reason, was_running=was_running)
        why = "root message deleted" if reason == "thread root deleted" else "`!end`"
        # After a deletion this is only visible if the thread survived as a tombstone.
        await self.hub.post(
            channel, thread_ts,
            f":wastebasket: Session `{session_id[:8]}` ended ({why}"
            f"{', running task interrupted' if was_running else ''}). This thread is no longer bound to it"
            f"{' and terminal mirroring stops until the session is reopened' if self.hub.mirror else ''}. "
            f"The transcript is kept: `!resume {session_id}` or `cd {escape(cwd)} && claude --resume {session_id}`.",
        )

    async def _dispatch(self, channel: str, ts: str, thread_ts: str | None, user: str, text: str,
                        files: list[dict[str, Any]] | None = None) -> None:
        try:
            if text.startswith("!"):
                cmd, _, arg = text[1:].partition(" ")
                handler = getattr(self, f"_cmd_{cmd.lower()}", None)
                if handler is None:
                    await self.hub.post(channel, thread_ts or ts, f"Unknown command `!{escape(cmd)}`. Try `!help`.")
                    return
                await handler(channel, ts, thread_ts, user, arg.strip())
                return
            await self._prompt(channel, ts, thread_ts, user, text, files or [])
        except Exception:
            log.exception("dispatch failed")
            await self.hub.post(channel, thread_ts or ts, ":x: internal error (see bot log)")

    async def _prompt(self, channel: str, ts: str, thread_ts: str | None, user: str, text: str,
                      files: list[dict[str, Any]] | None = None) -> None:
        sess: ThreadSession | None = None
        if thread_ts:
            sess = self.hub.get_session(channel, thread_ts)
            if sess is None:
                sess = self.hub.create_session(channel, thread_ts, self.cfg.default_cwd, user)
                await self.hub.post(channel, thread_ts, f"_No session was bound to this thread; started a new one in "
                                    f"{inline_code(str(self.cfg.default_cwd))}._")
        else:
            sess = self.hub.create_session(channel, ts, self.cfg.default_cwd, user)
        q = self.hub.question_for(sess.rec.key)
        if q is not None:
            open_questions = [i for i in range(len(q.questions)) if q.text_of(i) not in q.answers]
            if files:
                label = (text.strip() + " 📎").strip()
                text = await self._receive_files(sess, files, text, user, require_file=True)
                if not text:
                    return
                if q.future.done() or self.hub.questions.get(q.id) is not q:
                    await self.hub.post(channel, sess.rec.thread_ts, "Question closed during upload. Your answer was not submitted.")
                    return
                open_questions = [i for i in open_questions if q.text_of(i) not in q.answers]
                if len(open_questions) > 1:
                    q.attached_replies[ts] = (user, text, label)
                    await self.hub.post(channel, sess.rec.thread_ts, "Which question is this attachment for?", blocks=[
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Which question is this attachment for?"}},
                        {"type": "actions", "elements": [
                            button(f"{i+1}. {q.text_of(i)}"[:75], f"sac_qa_attach_{i}", f"{q.id}|{ts}|{i}")
                            for i in open_questions]}])
                    return
                accepted = bool(open_questions) and q.answer(open_questions[0], text, label=label)
            else:
                accepted = bool(text) and q.answer_free_text(text)
            if accepted:
                await self._refresh_answer(q, channel, sess.rec.thread_ts)
            return
        if self.hub.mirror and (lk := self.hub.mirror.locked(sess.rec.session_id)):
            if files:
                text = await self._receive_files(sess, files, text, user, require_file=True)
                if not text:
                    return
            await self.hub.offer_fork(sess.rec, user, text, self.hub.mirror.lock_text(lk))
            return
        if sess.queue.qsize() >= self.cfg.max_queued_per_thread:
            await self.hub.post(channel, sess.rec.thread_ts,
                                f":no_entry: {sess.queue.qsize()} messages are already waiting in this thread (limit "
                                f"{self.cfg.max_queued_per_thread}); this one was not queued. Wait, or `!stop` to "
                                "drop the queue.")
            return
        if files:
            text = await self._receive_files(sess, files, text, user)
            if not text:
                return
        ahead = sess.submit(text, user)
        if ahead:
            value = f"{sess.rec.key}|{sess.last_task_id}"
            await self.hub.post(channel, sess.rec.thread_ts, f"Queued ({ahead} ahead).", blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": f"_Queued: {ahead} ahead._\n{inline_code(policy.redact_log(text)[:120])}"}},
                {"type": "actions", "block_id": "sac_queue", "elements": [
                    button(":fast_forward: Interrupt and send now", "sac_q_now", value, "primary"),
                    button("Keep queued", "sac_q_keep", value),
                    button("Withdraw this message", "sac_q_withdraw", value)]},
            ])

    async def _refresh_answer(self, q: PendingQuestion, channel: str, thread: str) -> None:
        if not q.future.done() and q.ts:
            try:
                await self.hub.web.chat_update(channel=channel, ts=q.ts, text="Claude asks", blocks=q.blocks())
            except Exception:
                log.exception("failed to update question card")
        await self.hub.post(channel, thread, ":speech_balloon: Taken as your answer to Claude's question.")

    async def _receive_files(self, sess: ThreadSession, files: list[dict[str, Any]], text: str, user: str,
                             require_file: bool = False) -> str:
        """Download attached files, tell the thread what arrived, and return the prompt
        for Claude ("" if there is nothing to send)."""
        rec = sess.rec
        msg_ts = await self.hub.post(rec.channel, rec.thread_ts, f":paperclip: Receiving {len(files)} file(s)…")
        saved, problems = await self.hub.inbox.fetch(files, rec.session_id, self.hub.web)
        lines = []
        if saved:
            lines.append(":paperclip: Received: " + ", ".join(
                f"`{escape(s.name)}` ({s.kind}, {s.size / 1000:.0f} KB)" for s in saved))
        lines += [f":warning: {p}" for p in problems]
        if not saved and (require_file or not text):
            lines.append("_Nothing was sent to Claude._")
        note = "\n".join(lines) or "No files received."
        if msg_ts:
            try:
                await self.hub.web.chat_update(channel=rec.channel, ts=msg_ts, text=note)
            except Exception:
                await self.hub.post(rec.channel, rec.thread_ts, note)
        else:
            await self.hub.post(rec.channel, rec.thread_ts, note)
        self.hub.audit.write(event="inbox", session=rec.session_id, thread=rec.key, by=user,
                             files=[{"name": s.name, "kind": s.kind, "bytes": s.size, "path": str(s.path)} for s in saved],
                             problems=len(problems))
        if not saved:
            return "" if require_file else text
        return prompt_with_files(text, saved)

    # -- commands ----------------------------------------------------------
    async def _cmd_help(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        await self.hub.post(channel, thread_ts or ts, HELP.format(
            root=inline_code(str(self.cfg.cwd_roots[0])), timeout=int(self.cfg.approval_timeout_s // 60)))

    async def _thread_root(self, channel: str, ts: str, thread_ts: str | None, text: str) -> str | None:
        """Top-level command -> the command message is the thread root.
        Command inside a thread -> post a new top-level root message."""
        if thread_ts is None:
            return ts
        return await self.hub.post(channel, None, text)

    async def _cmd_new(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        dir_arg, _, prompt = arg.partition(" ")
        dir_arg = dict(self.cfg.projects).get(dir_arg.lower(), dir_arg)  # project alias -> directory
        cwd, err = self.hub.validate_cwd(dir_arg) if dir_arg else (self.cfg.default_cwd, "")
        if cwd is None:
            await self.hub.post(channel, thread_ts or ts, f":warning: {escape(err)}")
            return
        root = await self._thread_root(channel, ts, thread_ts, f":sparkles: New session in {inline_code(str(cwd))}")
        if root is None:
            return
        sess = self.hub.create_session(channel, root, cwd, user)
        if thread_ts is None:
            await self.hub.post(channel, root, f":sparkles: New session in {inline_code(str(cwd))} "
                                f"(`{sess.rec.session_id[:8]}`). Reply in this thread.")
        else:
            await self.hub.post(channel, thread_ts, f"Started a new session in {inline_code(str(cwd))} — see the "
                                "new thread in the channel.")
        if prompt.strip():
            sess.submit(prompt.strip(), user)

    async def _cmd_stop(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str,
                        from_button: bool = False) -> None:
        """`!stop` = stop the running reply and drop queued messages; `!stop bg` = stop background
        tasks; `!stop all` = both. Reports what actually happened."""
        what = arg.strip().lower()
        if what not in ("", "keep", "bg", "background", "all"):
            await self.hub.post(channel, thread_ts or ts, "Usage: `!stop` (reply + queue) · `!stop bg` "
                                "(background tasks) · `!stop all`")
            return
        if thread_ts:
            sess = self.hub.sessions.get(f"{channel}:{thread_ts}")
        else:
            busy = [s for s in self.hub.sessions.values()
                    if s.rec.channel == channel and (s.busy or s.bg_tasks or not s.queue.empty())]
            if len(busy) > 1:
                await self.hub.post(channel, ts, f"{len(busy)} sessions are active; send `!stop` inside the thread.")
                return
            sess = busy[0] if busy else None
        reply_to = thread_ts or ts
        if sess is None or not (sess.busy or sess.bg_tasks or not sess.queue.empty()):
            await self.hub.post(channel, reply_to, "Nothing is running here.")
            return
        if not thread_ts:
            reply_to = sess.rec.thread_ts
        msg_ts = None if from_button else await self.hub.post(channel, reply_to, ":stop_button: Stopping…")
        lines: list[str] = []
        if what in ("", "keep", "all"):
            r = await sess.stop(user, clear_queue=what != "keep")
            if r.was_running and r.stopped:
                lines.append(":stop_button: Stopped the running reply.")
            elif r.was_running:
                lines.append(":warning: Asked Claude to stop, but the reply is still running after "
                             f"{STOP_WAIT_S:.0f} s" + (f" ({escape(r.error)})" if r.error else "")
                             + ". `!exit force` ends the Claude Code process.")
            elif r.error:
                lines.append(f":warning: {escape(r.error)}")
            if r.dropped:
                lines.append(f":wastebasket: Dropped {r.dropped} queued message(s).")
        if what in ("bg", "background", "all") and sess.bg_tasks:
            stopped, failed = await sess.stop_background(user)
            if stopped:
                lines.append(":gear: Stopped background task(s): " + "; ".join(escape(d[:60]) for d in stopped))
            if failed:
                lines.append(":warning: Could not stop: " + "; ".join(escape(d[:60]) for d in failed)
                             + " (`!exit force` ends the process and its tasks)")
        if sess.bg_tasks:
            lines.append(f":gear: {len(sess.bg_tasks)} background task(s) still running: "
                         + "; ".join(escape(d[:60]) for d in sess.bg_tasks.values())
                         + ("" if what in ("bg", "background", "all") else " — `!stop bg` to stop them"))
        text = "\n".join(lines) or "Nothing was running."
        if msg_ts:
            try:
                await self.hub.web.chat_update(channel=channel, ts=msg_ts, text=text)
                return
            except Exception:
                log.exception("failed to update the stop message")
        await self.hub.post(channel, reply_to, text)

    async def _cmd_mode(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        sess = self.hub.get_session(channel, thread_ts) if thread_ts else None
        reply_to = thread_ts or ts
        if sess is None:
            await self.hub.post(channel, reply_to, f"Use `!mode` inside a session thread. New sessions start in "
                                f"`{self.cfg.default_mode}` mode (DEFAULT_PERMISSION_MODE).")
            return
        want = arg.strip().lower()
        if not want:
            await self.hub.post(channel, reply_to, f"This session is in `{sess.rec.mode}` mode.")
            return
        if want not in ("auto", "ask"):
            await self.hub.post(channel, reply_to, "Usage: `!mode auto` or `!mode ask`")
            return
        if sess.rec.mode == want:
            await self.hub.post(channel, reply_to, f"This session is already in `{want}` mode.")
            return
        await sess.set_mode(want)
        self.hub.audit.write(event="mode-change", session=sess.rec.session_id, thread=sess.rec.key, by=user, mode=want)
        msg = (":zap: *auto mode*: Claude Code's classifier approves routine actions. Still enforced: protected "
               "paths and credential access are blocked; commands that may dump the environment, send data "
               "off-node or touch protected paths still ask you here. Auto mode needs Sonnet/Opus."
               if want == "auto" else ":lock: *ask mode*: every action with side effects asks for approval here.")
        await self.hub.post(channel, reply_to, msg)

    async def _cmd_model(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        await self._model_effort(channel, ts, thread_ts, user, arg, "model")

    async def _cmd_effort(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        await self._model_effort(channel, ts, thread_ts, user, arg, "effort")

    async def _model_effort(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str,
                            what: str) -> None:
        sess = self.hub.get_session(channel, thread_ts) if thread_ts else None
        reply_to = thread_ts or ts
        usage = ("Usage: `!model opus|sonnet|haiku|<model id>|default`" if what == "model"
                 else f"Usage: `!effort {'|'.join(EFFORTS)}|default`")
        if sess is None:
            await self.hub.post(channel, reply_to, f"Use `!{what}` inside a session thread. Bot defaults: model "
                                f"`{self.cfg.model or 'Claude Code default'}`, effort "
                                f"`{self.cfg.default_effort or 'Claude Code default'}`.")
            return
        want = arg.strip()
        if not want:
            await self.hub.post(channel, reply_to, await self._model_effort_text(sess))
            return
        w = want.lower()
        if what == "effort":
            if w not in (*EFFORTS, "default"):
                await self.hub.post(channel, reply_to, usage)
                return
            when = sess.set_model_effort(effort=None if w == "default" else w, clear_effort=w == "default")
        else:
            ok = w == "default" or re.fullmatch(r"(opus|sonnet|haiku)(\[1m\])?", w) or (
                self.hub.bedrock_model_ok(want) if self.cfg.provider == "bedrock" else want.startswith("claude-"))
            if not ok or not re.fullmatch(r"[\w.:/\[\]-]+", want):
                await self.hub.post(channel, reply_to, usage)
                return
            when = sess.set_model_effort(model=None if w == "default" else want, clear_model=w == "default")
        self.hub.audit.write(event=f"{what}-change", session=sess.rec.session_id, thread=sess.rec.key, by=user,
                             value=w)
        await self.hub.post(channel, reply_to, f":gear: {what} → `{w}`, takes effect {when}.\n"
                            + await self._model_effort_text(sess))

    async def _model_effort_text(self, sess: ThreadSession) -> str:
        r = self.cfg
        model = sess.rec.model or r.model or "Claude Code default"
        effort = sess.rec.effort or r.default_effort or "not set (Claude Code default for the model)"
        lines = [f"*model* `{model}`" + (" _(this thread)_" if sess.rec.model else " _(bot default)_"),
                 f"*effort* `{effort}`" + (" _(this thread)_" if sess.rec.effort else " _(bot default)_")]
        used_model, used_effort = await asyncio.to_thread(sess.last_used)
        if used_model or used_effort:
            lines.append(f"_last reply in this session (Slack or terminal) used `{used_model or '?'}` · effort "
                         f"`{used_effort or '?'}`_")
        if sess._options_stale:
            lines.append("_pending: applies when Claude Code restarts for the next message_")
        return "\n".join(lines)

    async def _cmd_status(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        sess = self.hub.get_session(channel, thread_ts) if thread_ts else None
        await self.hub.post(channel, thread_ts or ts, await self.hub.status_text(sess))

    def _project_of(self, cwd: str) -> str | None:
        real = os.path.realpath(cwd)
        best = None
        for alias, d in self.cfg.projects:
            rd = os.path.realpath(d)
            if (real == rd or real.startswith(rd + "/")) and (best is None or len(rd) > len(best[1])):
                best = (alias, rd)
        return best[0] if best else None

    def _session_state(self, r: Any) -> str:
        s = self.hub.sessions.get(r.key)
        if self.hub.mirror and self.hub.mirror.locked(r.session_id):
            return ":lock: in terminal"
        if s and s.pending:
            return ":raised_hand: waiting for you"
        if s and s.busy:
            return ":large_green_circle: running"
        if s and s.bg_tasks:
            return f":gear: {len(s.bg_tasks)} background"
        if s and s.connected:
            return ":large_blue_circle: idle"
        return ":white_circle: closed"

    async def _cmd_sessions(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        recs = sorted(self.hub.store.records.values(), key=lambda r: r.last_active, reverse=True)[:15]
        titles = await asyncio.to_thread(_titles, [r.session_id for r in recs])
        blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn",
                                         "text": "*Sessions* (most recent first)" if recs else "_No sessions yet._"}}]
        lines = ["Sessions"]
        for r in recs:
            where = self._project_of(r.cwd) or _short_path(r.cwd)
            title = escape(_clip_line(titles.get(r.session_id) or "(untitled)", 70))
            text = (f"{self._session_state(r)} · *{title}*\n{inline_code(where)} · {r.turns} turns · "
                    f"{_ago(r.last_active)} · `{r.session_id[:8]}`")
            sec: dict[str, Any] = {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}
            link = await self._permalink(r.channel, r.thread_ts)
            if link:
                sec["accessory"] = {"type": "button", "action_id": f"sac_open_{r.session_id[:8]}", "url": link,
                                    "text": {"type": "plain_text", "text": "Open"}}
            blocks.append(sec)
            lines.append(f"{title} ({where})")
        if arg == "all":
            extra = await self._recent_terminal_sessions()
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": extra[:2900]}})
        await self.hub.post(channel, thread_ts or ts, "\n".join(lines)[:3000], blocks=blocks[:50])

    async def _recent_terminal_sessions(self) -> str:
        lines = ["*Recent Claude Code sessions on this node* (`!resume <id>` to attach)"]
        try:
            infos = await asyncio.to_thread(list_sessions, None, 10)
            for i in infos:
                lines.append(f"• `{i.session_id}` {inline_code(_short_path(i.cwd or '?'))} · "
                             f"{_ago(i.last_modified / 1000)} · {escape((i.summary or '')[:60])}")
        except Exception as e:
            lines.append(f"_failed to list: {escape(str(e))}_")
        return "\n".join(lines)

    async def _cmd_jobs(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        reply_to = thread_ts or ts
        if self.jobs is None:
            await self.hub.post(channel, reply_to, "Job tracking is off (JOBS_ENABLED=0, or no systemd user "
                                "services on this machine).")
            return
        ids = await asyncio.to_thread(self.jobs.recent, 15)
        if not ids:
            await self.hub.post(channel, reply_to, "No jobs yet. Claude starts them with `sac-job submit`; you can "
                                f"too, from a terminal: `{SAC_JOB} submit --gpus 1 -- python train.py`.")
            return
        blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Jobs* (running first)"}}]
        for jid in ids:
            line = await asyncio.to_thread(self.jobs.describe, jid)
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": inline_code(line[:150])},
                           "accessory": button("Log", "sac_job_log", jid)})
        await self.hub.post(channel, reply_to, "Jobs", blocks=blocks[:50])

    async def _cmd_projects(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        reply_to = thread_ts or ts
        if not self.cfg.projects:
            await self.hub.post(channel, reply_to, "No projects configured. Add e.g. "
                                "`PROJECTS=web=~/code/webapp,ml=~/research/model` to the env file.")
            return
        blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text":
                                         "*Projects* — tap to start a session there, or `!new <alias> <message>`"}}]
        for alias, d in self.cfg.projects:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text":
                           f"*{alias}* · {inline_code(_short_path(d))}"},
                           "accessory": button(":sparkles: New session", f"sac_proj_new_{alias}", alias)})
        await self.hub.post(channel, reply_to, "Projects", blocks=blocks[:50])

    async def _on_project_new(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        user = body.get("user", {}).get("id", "")
        if user not in self.cfg.allowed_users:
            log.warning("project click from non-allowlisted user %s ignored", user)
            return
        channel = body.get("channel", {}).get("id", "")
        alias = str(action.get("value", ""))
        d = dict(self.cfg.projects).get(alias)
        if not d or not channel:
            return
        msg = body.get("message", {})
        here = str(msg.get("thread_ts") or msg.get("ts") or "")
        # Like `!new <dir>` typed where the card is: a new top-level thread is opened for the session.
        await self._cmd_new(channel, here, here, user, d)

    async def _permalink(self, channel: str, ts: str) -> str | None:
        try:
            resp = await self.hub.web.chat_getPermalink(channel=channel, message_ts=ts)
            return resp.get("permalink")
        except Exception:
            return None

    async def _cmd_resume(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        sid = arg.strip()
        reply_to = thread_ts or ts
        if not _UUID.match(sid):
            await self.hub.post(channel, reply_to, "Usage: `!resume <session-uuid>` (see `!sessions all`)")
            return
        info = await asyncio.to_thread(get_session_info, sid)
        if info is None or not info.cwd:
            await self.hub.post(channel, reply_to, f"Session `{sid}` not found on this node.")
            return
        cwd, err = self.hub.validate_cwd(info.cwd)
        if cwd is None:
            await self.hub.post(channel, reply_to, f":warning: cannot attach: {escape(err)}")
            return
        if (old := self.hub.store.find_by_session(sid)) is not None:
            await self.hub.detach(old.key)
        root = await self._thread_root(channel, ts, thread_ts, f":link: Attached session `{sid[:8]}`")
        if root is None:
            return
        sess = self.hub.create_session(channel, root, cwd, user, session_id=sid)
        await self.hub.post(
            channel, root,
            f":link: Attached session `{sess.rec.session_id}` in {inline_code(str(cwd))}\n"
            f"> {escape((info.summary or '')[:200])}\n"
            "_Reply here to continue it. Don't drive the same session from the terminal at the same time._",
        )

    async def _cmd_import(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        reply_to = thread_ts or ts
        prefixes = arg.split()
        if not prefixes:
            await self.hub.post(channel, reply_to, "Usage: `!import <session-id-prefix> [...]` (see `!sessions all`)")
            return
        found, errors = await asyncio.to_thread(importer.resolve_ids, prefixes)
        lines = [f":warning: {e}" for e in errors]
        for info in found:
            sid = info.session_id
            if (old := self.hub.store.find_by_session(sid)) is not None:
                link = await self._permalink(old.channel, old.thread_ts)
                lines.append(f"`{sid[:8]}` is already bound to " + (f"<{link}|a thread>" if link else "a thread"))
                continue
            cwd, err = self.hub.validate_cwd(info.cwd or "")
            if cwd is None:
                lines.append(f":warning: `{sid[:8]}`: {escape(err)}")
                continue
            ex = await asyncio.to_thread(importer.exchanges, info)
            shown = ex[-5:]
            root = await self.hub.post(channel, None, importer.card_text(
                info, len(ex), len(shown), _ago(info.last_modified / 1000), self.cfg.node_name))
            if root is None:
                lines.append(f":x: `{sid[:8]}`: could not post to this channel")
                continue
            self.hub.create_session(channel, root, cwd, user, session_id=sid)
            for i, e in enumerate(shown, start=len(ex) - len(shown) + 1):
                await asyncio.sleep(1.1)  # stay under Slack's ~1 msg/s per channel
                await self.hub.post(channel, root, importer.format_exchange(e, i, len(ex)))
            self.hub.audit.write(event="import", session=sid, thread=f"{channel}:{root}", by=user,
                                 exchanges=len(shown))
            lines.append(f":white_check_mark: `{sid[:8]}` imported ({len(shown)} of {len(ex)} exchanges)")
        await self.hub.post(channel, reply_to, "\n".join(lines) or "Nothing to import.")

    async def _on_fork_send(self, body: dict[str, Any], action: dict[str, Any]) -> None:
        user = body.get("user", {}).get("id", "")
        channel = body.get("channel", {}).get("id", "")
        thread = body.get("message", {}).get("thread_ts", "")
        token = str(action.get("value", ""))
        if user not in self.cfg.allowed_users or not thread:
            return
        offer = self.hub.fork_offers.get(token)
        if offer is None:
            await self.hub.post(channel, thread, "This fork offer has expired or was already used. Check this thread for the new fork.")
            return
        key, sid, author, prompt = offer
        if user != author or key != thread_key(channel, thread):
            return
        del self.hub.fork_offers[token]  # claim before awaiting Slack
        sess = self.hub.get_session(channel, thread)
        if sess is None or sess.rec.session_id != sid:
            await self.hub.post(channel, thread, "The original session is no longer bound to this thread.")
            return
        if not await self._fork(sess, user, prompt):
            self.hub.fork_offers[token] = offer

    async def _cmd_fork(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        sess = self.hub.get_session(channel, thread_ts) if thread_ts else None
        if sess is None or not (sess.rec.has_history or sess.rec.fork_from):
            await self.hub.post(channel, thread_ts or ts, "Use `!fork` inside a session thread that has history.")
            return
        await self._fork(sess, user)

    async def _fork(self, sess: ThreadSession, user: str, prompt: str = "") -> bool:
        parent = sess.rec
        channel, thread_ts = parent.channel, parent.thread_ts
        title = f":twisted_rightwards_arrows: *Fork* of session `{parent.session_id[:8]}` in {inline_code(parent.cwd)}."
        detail = inline_code(policy.redact_log(prompt)[:240]) if prompt else "Reply here to continue from the same history."
        root = await self.hub.post(channel, None, f"{title}\n{detail}")
        if root is None:
            await self.hub.post(channel, thread_ts, "Could not create the fork. Please try again.")
            return False
        fork = self.hub.create_fork(channel, root, parent, user)
        if prompt:
            fork.submit(prompt, user)
        self.hub.audit.write(event="fork", session=parent.session_id, fork=fork.rec.session_id, by=user)
        link = await self._permalink(channel, root)
        await self.hub.post(channel, thread_ts, f":twisted_rightwards_arrows: Forked as `{fork.rec.session_id[:8]}`"
                            + (f" → <{link}|new thread>" if link else ""))
        return True

    async def _cmd_allowlist(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        reply_to = thread_ts or ts
        al = self.hub.allowlist
        sub, _, rest = arg.strip().partition(" ")
        sub = sub.lower()
        if sub in ("remove", "rm", "delete"):
            ent, msg = al.remove(rest)
            if ent is not None:
                self.hub.audit.write(event="allowlist-remove", id=ent.id, by=user, input=ent.command[:300])
                msg = f":wastebasket: Removed `{ent.id}` {inline_code(_clip_line(ent.command, 100))}. It asks again next time."
            await self.hub.post(channel, reply_to, msg)
            return
        if sub == "clear":
            n = al.clear()
            self.hub.audit.write(event="allowlist-clear", by=user, removed=n)
            await self.hub.post(channel, reply_to, f":wastebasket: Cleared {n} remembered command(s).")
            return
        if sub not in ("", "list"):
            await self.hub.post(channel, reply_to, "Usage: `!allowlist` · `!allowlist remove <id>` · `!allowlist clear`")
            return
        ents = sorted(al.entries.values(), key=lambda e: e.added, reverse=True)
        state = "on" if self.cfg.learn_approved_commands else "OFF (LEARN_APPROVED_COMMANDS=0)"
        lines = [f"*Remembered commands* ({len(ents)}; remembering is {state}). Exact command text + directory; "
                 "`!allowlist remove <id>` or `!allowlist clear`."]
        for e in ents[:20]:
            when = time.strftime("%m-%d", time.localtime(e.added))
            lines.append(f"`{e.id}` {inline_code(_clip_line(e.command, 100))} in "
                         f"{inline_code(_short_path(e.cwd))} · {when}")
        if len(ents) > 20:
            lines.append(f"_…and {len(ents) - 20} older ones_")
        if not ents:
            lines.append("_none yet: choose Run & remember here on a Bash approval to add a command_")
        lines.append("_Built-in read-only (any directory): " + ", ".join(f"`{c}`" for c in sorted(al.seeds)) + "_")
        lines.append("_Never remembered: commands with warnings, deletes/moves/kills, or shell expansion ($, *, ?)._")
        await self.hub.post(channel, reply_to, "\n".join(lines))

    async def _cmd_end(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        """End this thread's session: works for any root, including ones the bot posted (review R10)."""
        rec = self.hub.store.get(thread_key(channel, thread_ts)) if thread_ts else None
        if rec is None:
            await self.hub.post(channel, thread_ts or ts, "Use `!end` inside a session thread.")
            return
        sess = self.hub.sessions.get(rec.key)
        if sess and sess.bg_tasks and arg.strip().lower() != "force":
            await self.hub.post(channel, thread_ts, f":gear: {len(sess.bg_tasks)} background task(s) still run "
                                "here; ending the session stops them. Wait, `!stop bg`, or `!end force`.")
            return
        await self._end_session(rec.channel, rec.thread_ts, rec.session_id, rec.cwd, user, reason="!end")

    async def _cmd_exit(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        reply_to = thread_ts or ts
        rec = self.hub.store.get(thread_key(channel, thread_ts)) if thread_ts else None
        if rec is None:
            await self.hub.post(channel, reply_to, "Use `!exit` inside a session thread.")
            return
        resume = f"`cd {escape(rec.cwd)} && claude --resume {rec.session_id}`"
        sess = self.hub.sessions.get(rec.key)
        if sess is None or sess.worker is None or sess.worker.done():
            await self.hub.post(channel, reply_to, f"Nothing to close: no Claude Code process is running for this "
                                f"thread. The terminal can use the session: {resume}")
            return
        if sess.busy:
            await self.hub.post(channel, reply_to, "A reply is still running. Wait for it to finish, or `!stop` "
                                "it first, then `!exit`.")
            return
        if sess.bg_tasks and arg.strip().lower() != "force":
            tasks = "; ".join(escape(d[:60]) for d in sess.bg_tasks.values())
            await self.hub.post(channel, reply_to, f":gear: Background task(s) still running here: {tasks}. "
                                "Closing would kill them. Wait until they finish, or `!exit force`.")
            return
        killed = len(sess.bg_tasks)
        await sess.close()
        self.hub.audit.write(event="session-exit", session=rec.session_id, thread=rec.key, by=user,
                             killed_background_tasks=killed or None)
        await self.hub.post(channel, reply_to, ":wave: Closed this thread's Claude Code process"
                            + (f" ({killed} background task(s) stopped)" if killed else "")
                            + f". The terminal can use the session now: {resume}\n"
                            "_Reply here any time to continue from Slack (it reloads the session's history)._")

    async def _cmd_unlock(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        sess = self.hub.get_session(channel, thread_ts) if thread_ts else None
        if sess is None or self.hub.mirror is None:
            await self.hub.post(channel, thread_ts or ts, "Use `!unlock` inside a mirrored session thread.")
            return
        lk = self.hub.mirror.locked(sess.rec.session_id)
        if lk is None:
            await self.hub.post(channel, thread_ts, "This session is not locked by a terminal.")
            return
        if lk.pid and arg.strip().lower() != "force":
            # locked() already verified that this exact process is still alive.
            what = "background session" if lk.background else "terminal session"
            await self.hub.post(channel, thread_ts, (
                f"Not unlocked: the {what} (pid {lk.pid}) is still running, and Claude Code would refuse to "
                f"resume it here anyway. " + (f"On {self.cfg.node_name}: `claude stop {lk.session_id[:8]}` (conversation is kept); "
                                             if lk.background else "Exit it in the terminal (`/exit`); ")
                + "the lock is released automatically when it ends. `!unlock force` overrides."))
            return
        self.hub.mirror.unlock(sess.rec.session_id)
        self.hub.audit.write(event="terminal-unlock", session=sess.rec.session_id, by=user, pid=lk.pid)
        await self.hub.post(channel, thread_ts, ":unlock: Unlocked. If the terminal is in fact still open, don't "
                            "type in both places at once.")

    async def _cmd_upload(self, channel: str, ts: str, thread_ts: str | None, user: str, arg: str) -> None:
        reply_to = thread_ts or ts
        if not arg:
            await self.hub.post(channel, reply_to, "Usage: `!upload <path>`")
            return
        sess = self.hub.get_session(channel, thread_ts) if thread_ts else None
        base = sess.cwd if sess else self.cfg.default_cwd
        p = Path(arg).expanduser()
        p = (p if p.is_absolute() else base / p).resolve()
        err = await self.hub.upload_file(channel, reply_to, p, user)
        if err:
            await self.hub.post(channel, reply_to, f":warning: {escape(err)}")

    # -- lifecycle ---------------------------------------------------------
    async def run(self) -> None:
        handler = AsyncSocketModeHandler(self.app, self.cfg.slack_app_token)
        auth = await self.hub.web.auth_test()
        log.info("connected to Slack workspace %s as %s", auth.get("team"), auth.get("user"))
        await handler.connect_async()
        expiry = asyncio.create_task(self.hub.token_expiry_loop())
        cleaner = asyncio.create_task(cleanup_loop(self.hub.inbox))
        stalls = asyncio.create_task(self.hub.stall_loop())
        jobs_task = None
        if self.cfg.jobs_enabled:
            self.jobs = JobWatcher(self)
            jobs_task = asyncio.create_task(self.jobs.loop())
        if self.cfg.mirror_dirs:
            from .mirror import Mirror

            self.hub.mirror = Mirror(self.hub)
            await self.hub.mirror.start()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        try:
            n = await self.hub.report_unfinished()
            if n:
                log.info("reported %d unfinished message(s) from before the restart", n)
        except Exception:
            log.exception("could not report unfinished messages")
        log.info("slack-agent-controller ready: %d allowed users, roots=%s", len(self.cfg.allowed_users),
                 ",".join(str(r) for r in self.cfg.cwd_roots))
        await stop.wait()
        log.info("shutting down")
        expiry.cancel()
        cleaner.cancel()
        stalls.cancel()
        if jobs_task:
            jobs_task.cancel()
        if self.hub.mirror:
            await self.hub.mirror.stop()
        await self.hub.shutdown()
        await handler.close_async()


def _ago(t: float) -> str:
    d = max(0, int(time.time() - t))
    if d < 3600:
        return f"{d // 60}m ago"
    if d < 86400:
        return f"{d // 3600}h ago"
    return f"{d // 86400}d ago"


def _titles(session_ids: list[str]) -> dict[str, str]:
    """Readable titles (Claude Code's summary/custom title) for session ids, best effort."""
    out: dict[str, str] = {}
    for sid in session_ids:
        try:
            info = get_session_info(sid)
        except Exception:
            info = None
        if info is not None:
            out[sid] = info.custom_title or info.summary or ""
    return out


def _clip_line(s: str, n: int) -> str:
    first = s.strip().splitlines()[0] if s.strip() else ""
    more = len(s.strip().splitlines()) > 1
    return first[: n - 1] + "…" if len(first) > n or more else first


def _short_path(p: str) -> str:
    home = str(Path.home())
    return "~" + p[len(home):] if p.startswith(home) else p

