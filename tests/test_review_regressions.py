"""Findings of the 2026-11-17 code review (an internal security review, 2026-11), as regression tests of the
behavior the product must have. The originals asserted that the defect existed.

R1  read-only Bash auto-allow could be bypassed (`sort -o..`, `cat .e*`, Grep glob=.env)
R2  "Approve all" overrode must-ask commands            -> tests/test_approvals.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from slack_agent_controller import policy
from slack_agent_controller.hub import Hub
from slack_agent_controller.mirror import Mirror
from tests.helpers import FakeWeb, make_config


def test_r1_sort_output_option_is_not_auto_allowed(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    cfg = make_config(tmp_path, [work])
    assert policy.decide(cfg, "Bash", {"command": "sort -o../outside.txt input.txt"}, work).decision == "ask"


def test_r1_wildcard_to_dotenv_is_not_auto_allowed_and_goes_to_slack(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env").write_text("DUMMY_PRIVATE_VALUE=review-only\n")
    cfg = make_config(tmp_path, [work])
    assert policy.decide(cfg, "Bash", {"command": "cat .e*"}, work).decision == "ask"
    sess = Hub(cfg, FakeWeb()).create_session("D1", "1.0", work, "UOWNER")  # type: ignore[arg-type]
    sess.rec.mode = "auto"
    inp = {"tool_name": "Bash", "tool_input": {"command": "cat .e*"}, "cwd": str(work), "tool_use_id": "t1"}
    out = asyncio.run(sess._pre_tool_hook(inp, "t1", None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_r1_grep_glob_selecting_dotenv_is_checked(tmp_path: Path):
    cfg = make_config(tmp_path, [tmp_path])
    inp = {"pattern": ".", "glob": ".env", "path": str(tmp_path / "project")}
    res = policy.decide(cfg, "Grep", inp, tmp_path)
    assert res.decision == "ask" and res.force_slack
    assert policy.output_protected(cfg, "Grep", inp, tmp_path)


# ---------------------------------------------------------------------------
# Batch 1: R4 (final update), R5, R6, R7, R8, R10
# ---------------------------------------------------------------------------
import contextlib  # noqa: E402
import time  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

import slack_agent_controller.session as session_mod  # noqa: E402
from slack_agent_controller.app import SlackFrontend  # noqa: E402
from slack_agent_controller.session import Prompt, TurnRenderer, _Turn  # noqa: E402
from slack_agent_controller.slackfmt import fit_code_block  # noqa: E402


def _sess(tmp: Path):
    hub = Hub(make_config(tmp, [tmp]), FakeWeb())  # type: ignore[arg-type]
    return hub.create_session("D1", "1.0", tmp, "UOWNER")


def _start_turn(s, tmp: Path) -> None:
    s._begin(_Turn(TurnRenderer(s.hub, "D1", "1.0", tmp), "UOWNER", time.monotonic(), False))


def test_r6_stop_drops_queued_prompts_and_reports(tmp_path: Path):
    async def run():
        s = _sess(tmp_path)
        s.client = AsyncMock()
        _start_turn(s, tmp_path)
        s.queue.put_nowait(Prompt("queued action", "UOWNER"))
        s.queue.put_nowait(Prompt("another", "UOWNER"))

        async def interrupted():
            await s._end_turn()   # the CLI ends the turn when interrupted

        s.client.interrupt.side_effect = interrupted
        r = await s.stop("UOWNER", wait_s=2)
        assert r.was_running and r.stopped and r.dropped == 2 and r.error is None
        assert s.queue.empty()
    asyncio.run(run())


def test_r6_stop_reports_interrupt_failure(tmp_path: Path):
    async def run():
        s = _sess(tmp_path)
        s.client = AsyncMock()
        s.client.interrupt.side_effect = RuntimeError("boom")
        _start_turn(s, tmp_path)
        r = await s.stop("UOWNER", wait_s=0.3)
        assert r.was_running and not r.stopped and r.error and "RuntimeError" in r.error
    asyncio.run(run())


def test_r6_stop_command_reports_background_tasks_and_stops_them(tmp_path: Path):
    async def run():
        f = SlackFrontend(make_config(tmp_path, [tmp_path]))
        web = FakeWeb()
        f.hub.web = web  # type: ignore[assignment]
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.client = AsyncMock()
        s.bg_tasks["task1"] = "background calculation"
        await f._cmd_stop("D1", "2.0", "1.0", "UOWNER", "")
        text = web.texts()[-1]
        assert "background calculation" in text and "!stop bg" in text, text
        s.client.stop_task.assert_not_awaited()

        async def killed(tid):
            s.bg_tasks.pop(tid, None)

        s.client.stop_task.side_effect = killed
        await f._cmd_stop("D1", "2.1", "1.0", "UOWNER", "bg")
        s.client.stop_task.assert_awaited_once_with("task1")
        assert "Stopped background task" in web.texts()[-1] and not s.bg_tasks
        await f._cmd_stop("D1", "2.2", "1.0", "UOWNER", "")
        assert "Nothing is running" in web.texts()[-1]
    asyncio.run(run())


def test_r5_queued_prompt_rechecks_terminal_lock(tmp_path: Path):
    async def run():
        s = _sess(tmp_path)

        class Lock:
            pid, opened_at, background, session_id = 4242, 0.0, False, s.rec.session_id

        class LockedMirror:
            def locked(self, sid):
                return Lock()

            def lock_text(self, st):
                return ":lock: open in a terminal"

        s.hub.mirror = LockedMirror()  # type: ignore[assignment]
        s.client = AsyncMock()
        s._ensure_client = AsyncMock()  # type: ignore[method-assign]
        await s._run_prompt(Prompt("already queued", "UOWNER"))
        s.client.query.assert_not_awaited()
        assert any("queued message was not sent" in t for t in s.hub.web.texts())
    asyncio.run(run())


def test_r4_final_update_is_retried_then_posted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(session_mod, "FINAL_RETRY_DELAYS", (0.0, 0.0))

    async def run():
        s = _sess(tmp_path)
        r = TurnRenderer(s.hub, "D1", "1.0", tmp_path)
        r.ts = "2.0"
        r.commit_text("completed result")
        s.hub.web.chat_update = AsyncMock(side_effect=[OSError("outage"), {"ok": True}])
        await r.finish("done")
        assert s.hub.web.chat_update.await_count == 2 and "completed result" in r._last_sent

        r2 = TurnRenderer(s.hub, "D1", "1.0", tmp_path)
        r2.ts = "3.0"
        r2.commit_text("second result")
        s.hub.web.chat_update = AsyncMock(side_effect=OSError("outage"))
        await r2.finish("done")
        assert any("second result" in t for t in s.hub.web.texts()), "fallback: posted as a new message"
    asyncio.run(run())


def test_r7_live_reply_is_redacted(tmp_path: Path):
    s = _sess(tmp_path)
    r = TurnRenderer(s.hub, "D1", "1.0", tmp_path)
    token = "xo" + "xb-" + "1234567890-FAKE-REVIEW-TOKEN"   # assembled so secret scanners stay quiet
    r.delta("Output contains " + "xo" + "xb-" + "1234567890-")     # a token split across stream chunks
    r.delta("FAKE-REVIEW-TOKEN and more")
    assert token not in r.render()
    r.commit_text("Output contains " + token)
    for out in (r.render(), r.raw_markdown(), r.final_text()):
        assert token not in out and "<redacted>" in out


def test_r8_long_write_attaches_full_content_and_mentions_owner(tmp_path: Path):
    async def run():
        s = _sess(tmp_path)
        text = "ok\n" * 60 + "UNREVIEWED_LAST_LINE"
        tin = {"file_path": str(tmp_path / "x.txt"), "content": text}

        async def answer_later():
            for _ in range(50):
                if s.hub.approvals:
                    aid = next(iter(s.hub.approvals))
                    s.hub.resolve_approval(aid, "deny", "UOWNER")
                    return
                await asyncio.sleep(0.02)

        from claude_agent_sdk import ToolPermissionContext
        t = asyncio.create_task(answer_later())
        await s.hub.request_approval(s, "Write", tin, ToolPermissionContext(), "file modification")
        await t
        posts = [kw for op, kw in s.hub.web.calls if op == "post"]
        uploads = [kw for op, kw in s.hub.web.calls if op == "upload"]
        assert posts[0]["text"].startswith("<@UOWNER>")
        assert "more line(s) in the attached file" in json_dump(posts[0]["blocks"])
        assert uploads and "UNREVIEWED_LAST_LINE" in uploads[0]["content"]

        s.hub.web.calls.clear()
        short = {"file_path": str(tmp_path / "y.txt"), "content": "one line\n"}
        t = asyncio.create_task(answer_later())
        await s.hub.request_approval(s, "Write", short, ToolPermissionContext(), "file modification")
        await t
        assert not [kw for op, kw in s.hub.web.calls if op == "upload"], "nothing hidden -> no attachment"
    asyncio.run(run())


def json_dump(x) -> str:
    import json
    return json.dumps(x)


def test_r8_escaped_block_stays_under_slack_limit():
    block, cut = fit_code_block("&" * 1000)
    assert len(block) <= 3000 and cut
    assert "&am```" not in block and not block.rstrip("`").rstrip().endswith("&am")
    small, cut2 = fit_code_block("a < b & c")
    assert small == "```a &lt; b &amp; c```" and not cut2


def test_r10_end_command_ends_bot_created_thread(tmp_path: Path):
    async def run():
        f = SlackFrontend(make_config(tmp_path, [tmp_path]))
        f.hub.web = FakeWeb()  # type: ignore[assignment]
        f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        # deleting a bot-authored root is still ignored (no author check bypass) ...
        await f._on_message({"type": "message", "subtype": "message_deleted", "channel_type": "im",
                             "channel": "D1", "deleted_ts": "1.0",
                             "previous_message": {"user": "UBOT", "bot_id": "BBOT", "ts": "1.0"}})
        for t in list(f._tasks):
            with contextlib.suppress(Exception):
                await t
        assert f.hub.store.get("D1:1.0") is not None
        # ... but `!end` in the thread ends it
        await f._cmd_end("D1", "2.0", "1.0", "UOWNER", "")
        assert f.hub.store.get("D1:1.0") is None
        assert any("ended (`!end`" in t for t in f.hub.web.texts())
        audit = (f.cfg.state_dir / "audit.jsonl").read_text()
        assert '"reason": "!end"' in audit
    asyncio.run(run())


def test_r3_first_turn_crash_resumes_existing_transcript(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    s = _sess(tmp_path)
    assert s._options().resume is None                      # brand new: no transcript yet
    d = tmp_path / ".claude" / "projects" / "slug"
    d.mkdir(parents=True)
    (d / f"{s.rec.session_id}.jsonl").write_text('{"type":"user"}\n')   # first turn started, then crashed
    restored = Hub(s.cfg, FakeWeb()).get_session("D1", "1.0")          # bot restart
    assert restored is not None
    o = restored._options()
    assert o.resume == s.rec.session_id and o.session_id is None
    assert restored.rec.has_history


def test_r4_mirror_upload_failure_is_kept_and_retried(tmp_path: Path):
    from tests.test_mirror import Env, _asst, _user

    async def run():
        e = Env(tmp_path)
        m = e.mirror
        await m.handle(e.ev("SessionStart", source="startup"))
        await m.handle(e.ev("UserPromptSubmit", prompt="hello"))
        e.append([_user("hello", 0), _asst([{"type": "text", "text": "answer"}], 1)])
        st = m.states[e.sid]
        e.web.files_upload_v2 = AsyncMock(side_effect=OSError("simulated upload outage"))
        await m._flush_turn(st)
        assert st.turns == 1 and len(st.pending_uploads) == 1          # the turn is done, its log is queued
        assert Path(st.pending_uploads[0]["path"]).exists()
        assert any("keeps it and retries" in t for t in e.web.texts())

        restored = Mirror(e.hub)                                        # survives a bot restart
        rst = restored.states[e.sid]
        assert len(rst.pending_uploads) == 1
        e.web.files_upload_v2 = AsyncMock(return_value={"ok": True})
        await restored._send_pending(rst)
        e.web.files_upload_v2.assert_awaited_once()
        kw = e.web.files_upload_v2.await_args.kwargs
        assert kw["filename"] == "turn-001.md" and "answer" in kw["content"]
        assert not rst.pending_uploads and not list((e.cfg.state_dir / "mirror_outbox").iterdir())
        assert any("attached `turn-001.md` (delayed)" in t for t in e.web.texts())
    asyncio.run(run())
