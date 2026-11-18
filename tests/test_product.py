"""Phone-oriented features: reply buttons (1), queue choice (2), Claude's questions (3),
projects and session list (4), final answer as its own message (5), stall notice (7)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.config import ConfigError, _projects
from slack_agent_controller.hub import Hub
from slack_agent_controller.inbox import Saved
from slack_agent_controller.session import (
    Prompt,
    StopReport,
    ThreadSession,
    TurnRenderer,
    _Turn,
)
from slack_agent_controller.slackfmt import text_blocks
from tests.helpers import FakeWeb, make_config


def _front(tmp: Path, **over: Any) -> SlackFrontend:
    f = SlackFrontend(make_config(tmp, [tmp], **over))
    f.hub.web = FakeWeb()  # type: ignore[assignment]
    return f


def _texts(f: SlackFrontend) -> list[str]:
    return f.hub.web.texts()  # type: ignore[attr-defined]


def _click(f: SlackFrontend, action_id: str, value: str, user: str = "UOWNER", **extra: Any) -> tuple[dict, dict]:
    body = {"user": {"id": user}, "channel": {"id": "D1"}, "message": {"ts": extra.pop("msg_ts", "50.0")}}
    return body, {"action_id": action_id, "value": value, **extra}


# -- (1) buttons under replies ----------------------------------------------------
def test_text_blocks_split_and_keep_fences_balanced():
    text = "intro\n```\n" + "\n".join(f"line {i}" for i in range(600)) + "\n```\nend"
    blocks = text_blocks(text, limit=1000, block_id="sac_reply")
    assert blocks[0]["block_id"] == "sac_reply"
    assert all(len(b["text"]["text"]) <= 1000 for b in blocks)
    assert all(b["text"]["text"].count("```") % 2 == 0 for b in blocks)
    assert "".join(b["text"]["text"] for b in blocks).count("line 599") == 1


def test_reply_shows_stop_while_running_and_handover_buttons_when_done(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        r = TurnRenderer(f.hub, "D1", "1.0", tmp_path)
        r.controls = s.controls
        r.ts = "2.0"
        r.commit_text("working on it")
        await r._update(final=False)
        ids = [e["action_id"] for b in f.hub.web.calls[-1][1]["blocks"] if b["type"] == "actions" for e in b["elements"]]
        assert ids == ["sac_ctl_stop"]
        s.bg_tasks["t"] = "watch"
        await r.finish("done")
        ids = [e["action_id"] for b in f.hub.web.calls[-1][1]["blocks"] if b["type"] == "actions" for e in b["elements"]]
        assert ids == ["sac_ctl_exit", "sac_ctl_mode_auto", "sac_ctl_status", "sac_ctl_stopbg"]
    asyncio.run(run())


def test_control_buttons_run_the_commands_and_check_the_user(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        await f._on_control(*_click(f, "sac_ctl_mode_auto", "D1:1.0"))
        assert s.rec.mode == "auto"
        await f._on_control(*_click(f, "sac_ctl_mode_auto", "D1:1.0"))
        assert s.rec.mode == "auto"
        await f._on_control(*_click(f, "sac_ctl_mode", "D1:1.0", text={"text": "Switch to ask mode"}))
        assert s.rec.mode == "ask"
        await f._on_control(*_click(f, "sac_ctl_stop", "D1:1.0"))
        assert "Nothing is running" in _texts(f)[-1]
        await f._on_control(*_click(f, "sac_ctl_mode_auto", "D1:1.0", user="UEVIL"))
        assert s.rec.mode == "ask", "non-allowlisted click ignored"
        assert '"control-click-rejected"' in (f.cfg.state_dir / "audit.jsonl").read_text()
        await f._on_control(*_click(f, "sac_ctl_exit", "D1:9.9"))
        assert "no longer bound" in _texts(f)[-1]
    asyncio.run(run())


# -- (2) queued message: interrupt now or keep -----------------------------------------
def test_queue_notice_offers_interrupt_and_it_runs_next(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.worker = asyncio.create_task(asyncio.sleep(10))      # a reply is "running"
        s._turn = object()  # type: ignore[assignment]
        await f._prompt("D1", "3.0", "1.0", "UOWNER", "first queued")
        await f._prompt("D1", "4.0", "1.0", "UOWNER", "fix the typo instead")
        notice = [kw for op, kw in f.hub.web.calls if op == "post" and kw.get("blocks")][-1]
        btn = notice["blocks"][1]["elements"][0]
        assert btn["action_id"] == "sac_q_now"
        stopped: list[bool] = []

        async def fake_stop(user, clear_queue=True, wait_s=15):  # noqa: ANN001
            stopped.append(clear_queue)
            s._turn = None
            from slack_agent_controller.session import StopReport
            return StopReport(True, True, None, 0, 0)

        s.stop = fake_stop  # type: ignore[method-assign]
        await f._on_queue_choice(*_click(f, "sac_q_now", btn["value"]))
        assert stopped == [False], "interrupting keeps the queue"
        first = s.queue.get_nowait()
        assert isinstance(first, Prompt) and first.text == "fix the typo instead"
        assert "runs now" in f.hub.web.calls[-1][1]["text"]
        await f._on_queue_choice(*_click(f, "sac_q_now", btn["value"]))   # second click: still queued -> ok
        s.worker.cancel()
    asyncio.run(run())


def test_queue_keep_and_stale_clicks(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.worker = asyncio.create_task(asyncio.sleep(10))
        s._turn = object()  # type: ignore[assignment]
        await f._prompt("D1", "3.0", "1.0", "UOWNER", "later")
        value = f"D1:1.0|{s.last_task_id}"
        await f._on_queue_choice(*_click(f, "sac_q_keep", value))
        assert "Kept in the queue" in f.hub.web.calls[-1][1]["text"]
        f.hub.tasks.done(s.last_task_id)
        await f._on_queue_choice(*_click(f, "sac_q_now", value))
        assert "no longer waiting" in f.hub.web.calls[-1][1]["text"]
        s.worker.cancel()
    asyncio.run(run())


# -- (3) Claude's questions -------------------------------------------------------------
QUESTIONS = [
    {"question": "Which functional?", "header": "Functional", "multiSelect": False,
     "options": [{"label": "PBE", "description": "fast"}, {"label": "r2SCAN", "description": "accurate"}]},
    {"question": "Which outputs?", "header": "Outputs", "multiSelect": True,
     "options": [{"label": "Energies", "description": ""}, {"label": "Forces", "description": ""},
                 {"label": "Stress", "description": ""}]},
]


async def _ask(f: SlackFrontend, s, questions=QUESTIONS):  # noqa: ANN001
    task = asyncio.create_task(f.hub.ask_questions(s, {"questions": questions}))
    for _ in range(50):
        if f.hub.questions:
            break
        await asyncio.sleep(0.01)
    q = next(iter(f.hub.questions.values()))
    return task, q


def test_questions_answered_with_buttons_and_checkboxes(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        task, q = await _ask(f, s)
        card = [kw for op, kw in f.hub.web.calls if op == "post"][-1]
        assert card["text"].startswith("<@UOWNER>")
        aids = [e["action_id"] for b in card["blocks"] if b["type"] == "actions" for e in b["elements"]]
        assert aids == ["sac_qa_pick_0_0", "sac_qa_pick_0_1", "sac_qa_multi_1", "sac_qa_submit_1", "sac_qa_skip"]
        assert q.id in s.pending
        await f._on_question(*_click(f, "sac_qa_pick_0_1", f"{q.id}|0|1"))
        await f._on_question(*_click(f, "sac_qa_submit_1", f"{q.id}|1"))      # nothing ticked yet
        assert not task.done()
        await f._on_question(*_click(f, "sac_qa_multi_1", "", selected_options=[
            {"value": f"{q.id}|1|0"}, {"value": f"{q.id}|1|2"}]))
        await f._on_question(*_click(f, "sac_qa_submit_1", f"{q.id}|1"))
        res = await asyncio.wait_for(task, 2)
        assert res.behavior == "allow"
        assert res.updated_input == {"questions": QUESTIONS,
                                     "answers": {"Which functional?": "r2SCAN", "Which outputs?": ["Energies", "Stress"]}}
        assert not f.hub.questions and q.id not in s.pending
        final = f.hub.web.calls[-1][1]
        assert "answered" in final["text"] and not [b for b in final["blocks"] if b["type"] == "actions"]
    asyncio.run(run())


def test_question_answered_by_a_thread_reply(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        task, q = await _ask(f, s, QUESTIONS[:1])
        s.submit = AsyncMock()  # type: ignore[method-assign]
        await f._prompt("D1", "5.0", "1.0", "UOWNER", "use HSE06 please")
        res = await asyncio.wait_for(task, 2)
        assert res.updated_input["answers"] == {"Which functional?": "use HSE06 please"}
        s.submit.assert_not_called()
        assert any("Taken as your answer" in t for t in _texts(f))
    asyncio.run(run())


def test_question_skip_stop_timeout_and_strangers(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        task, q = await _ask(f, s, QUESTIONS[:1])
        await f._on_question(*_click(f, "sac_qa_pick_0_0", f"{q.id}|0|0", user="UEVIL"))
        assert not task.done()
        await f._on_question(*_click(f, "sac_qa_skip", q.id))
        res = await asyncio.wait_for(task, 2)
        assert res.behavior == "allow" and "skipped" in res.updated_input["response"]

        task, q = await _ask(f, s, QUESTIONS[:1])
        s._turn = object()  # type: ignore[assignment]
        s.client = AsyncMock()
        await s.stop("UOWNER", wait_s=0)
        res = await asyncio.wait_for(task, 2)
        assert res.behavior == "deny" and res.interrupt
        s._turn = None

        f.hub.cfg = dataclasses.replace(f.hub.cfg, approval_timeout_s=0.1)
        task, q = await _ask(f, s, QUESTIONS[:1])
        res = await asyncio.wait_for(task, 2)
        assert res.behavior == "deny" and "No answer" in res.message
        await f._on_question(*_click(f, "sac_qa_pick_0_0", f"{q.id}|0|0"))     # late click
        assert "no longer open" in json.dumps(f.hub.web.calls[-1][1]["blocks"])
    asyncio.run(run())


def test_questions_always_reach_slack_even_in_auto_mode(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.rec.mode = "auto"
        s.allow_all.add("AskUserQuestion")
        out = await s._pre_tool_hook({"tool_name": "AskUserQuestion", "tool_input": {"questions": QUESTIONS},
                                      "cwd": str(tmp_path)}, "t1", None)
        assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
        assert "AskUserQuestion" not in f.cfg.disallowed_tools
    asyncio.run(run())


# -- (4) projects and the session list ---------------------------------------------------
def test_projects_config(tmp_path: Path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.setenv("PROJECTS", f"ml={tmp_path / 'a'}, web={tmp_path / 'b'}")
    assert _projects((tmp_path,)) == (("ml", str(tmp_path / "a")), ("web", str(tmp_path / "b")))
    for bad in ("x", "Bad Alias=/tmp", f"a={tmp_path / 'missing'}", "a=/etc", f"a={tmp_path / 'a'},a={tmp_path / 'b'}"):
        monkeypatch.setenv("PROJECTS", bad)
        with pytest.raises(ConfigError):
            _projects((tmp_path,))


def test_new_with_alias_and_project_button(tmp_path: Path):
    async def run():
        proj = tmp_path / "ml_dir"
        proj.mkdir()
        f = _front(tmp_path, projects=(("ml", str(proj)),))
        await f._cmd_new("D1", "1.0", None, "UOWNER", "ml")
        rec = f.hub.store.get("D1:1.0")
        assert rec is not None and rec.cwd == str(proj)
        await f._cmd_projects("D1", "2.0", None, "UOWNER", "")
        card = [kw for op, kw in f.hub.web.calls if op == "post"][-1]
        btn = card["blocks"][1]["accessory"]
        assert btn["action_id"] == "sac_proj_new_ml" and btn["value"] == "ml"
        n = len(f.hub.store.records)
        await f._on_project_new({"user": {"id": "UOWNER"}, "channel": {"id": "D1"}, "message": {"ts": "2.0"}}, btn)
        assert len(f.hub.store.records) == n + 1
        await f._on_project_new({"user": {"id": "UEVIL"}, "channel": {"id": "D1"}, "message": {"ts": "2.0"}}, btn)
        assert len(f.hub.store.records) == n + 1
    asyncio.run(run())


def test_sessions_list_has_titles_states_and_open_buttons(tmp_path: Path, monkeypatch):
    import slack_agent_controller.app as app_mod
    monkeypatch.setattr(app_mod, "_titles", lambda ids: {i: "Fix the phonon workflow" for i in ids})

    async def run():
        proj = tmp_path / "ml_dir"
        proj.mkdir()
        f = _front(tmp_path, projects=(("ml", str(proj)),))
        a = f.hub.create_session("D1", "1.0", proj, "UOWNER")
        f.hub.create_session("D1", "2.0", tmp_path, "UOWNER")
        a.pending.add("x")
        await f._cmd_sessions("D1", "3.0", None, "UOWNER", "")
        card = f.hub.web.calls[-1][1]
        body = json.dumps(card["blocks"])
        assert "Fix the phonon workflow" in body and "waiting for you" in body and "`ml`" in body
        assert sum(1 for b in card["blocks"] if b.get("accessory", {}).get("url")) == 2
    asyncio.run(run())


# -- (5) final answer as its own message when the reply was cut ----------------------------
def test_long_reply_posts_final_answer_separately(tmp_path: Path):
    async def run():
        hub = Hub(make_config(tmp_path, [tmp_path]), FakeWeb())  # type: ignore[arg-type]
        r = TurnRenderer(hub, "D1", "1.0", tmp_path)
        r.ts = "2.0"
        r.commit_text("step\n" * 2000)
        r.commit_text("**Conclusion:** use r2SCAN, the PBE results are off by 40 meV/atom.")
        await r.finish("done")
        posts = [kw["text"] for op, kw in hub.web.calls if op == "post"]
        assert any(p.startswith(":pushpin: *Final answer*") and "use r2SCAN" in p for p in posts), posts
        short = TurnRenderer(hub, "D1", "1.0", tmp_path)
        short.ts = "3.0"
        n = len(hub.web.calls)
        short.commit_text("short answer")
        await short.finish("done")
        assert not any(op == "post" for op, _ in hub.web.calls[n:]), "short replies are not duplicated"
    asyncio.run(run())


# -- (7) stall notice ------------------------------------------------------------------------
def test_stall_notice_once_and_not_while_waiting_for_you(tmp_path: Path):
    async def run():
        hub = Hub(dataclasses.replace(make_config(tmp_path, [tmp_path]), stall_notify_s=60), FakeWeb())  # type: ignore[arg-type]
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s._begin(_Turn(TurnRenderer(hub, "D1", "1.0", tmp_path), "UOWNER", time.monotonic(), False))
        s.last_event -= 30
        assert await hub.check_stalls() == 0                       # not yet
        s.last_event -= 100
        s.open_tools["t1"] = "Bash: python train.py"
        assert await hub.check_stalls() == 1
        assert "python train.py" in hub.web.texts()[-1] and "<@UOWNER>" in hub.web.texts()[-1]  # type: ignore[attr-defined]
        assert await hub.check_stalls() == 0                       # only once per quiet spell
        s.stall_notified = False
        s.pending.add("approval")                                  # waiting for the user is not a stall
        assert await hub.check_stalls() == 0
    asyncio.run(run())


def test_withdraw_one_and_reject_stale_clear(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.worker = asyncio.create_task(asyncio.sleep(10))
        s._turn = object()  # type: ignore[assignment]
        for text in ("first", "second"):
            await f._prompt("D1", "2.0", "1.0", "UOWNER", text)
        notice = f.hub.web.calls[-1][1]["blocks"]
        assert "second" in notice[0]["text"]["text"]
        withdraw = notice[1]["elements"][-1]
        await f._on_control(*_click(f, "sac_ctl_stop", s.rec.key))
        keep, clear = f.hub.web.calls[-1][1]["blocks"][-1]["elements"]
        assert "2" in clear["text"]["text"]
        await f._on_queue_choice(*_click(f, withdraw["action_id"], withdraw["value"]))
        assert s.queue.qsize() == 1
        await f._on_control(*_click(f, clear["action_id"], clear["value"]))
        assert "Queue changed" in f.hub.web.calls[-1][1]["text"]
        assert len(f.hub.tasks.records) == 1
        s.stop = AsyncMock(return_value=StopReport(True, True, None, 0, 0))
        await f._on_control(*_click(f, keep["action_id"], keep["value"]))
        s.stop.assert_awaited_once_with("UOWNER", clear_queue=False)
        await f._on_control(*_click(f, "sac_ctl_stop", s.rec.key))
        clear = f.hub.web.calls[-1][1]["blocks"][-1]["elements"][1]
        await f._on_control(*_click(f, clear["action_id"], clear["value"]))
        s.stop.assert_awaited_with("UOWNER", clear_queue=True)
        s.worker.cancel()
    asyncio.run(run())


@pytest.mark.parametrize("count", [1, 2])
def test_attachment_goes_to_selected_answer(tmp_path: Path, count: int):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        task, q = await _ask(f, s, QUESTIONS[:count])
        path = tmp_path / "plot.png"
        f.hub.inbox.fetch = AsyncMock(return_value=([Saved(path.name, path, "image", 200)], []))
        await f._prompt("D1", "4.0", "1.0", "UOWNER", "see plot", [{"id": "F1"}])
        if count == 2:
            action = f.hub.web.calls[-1][1]["blocks"][-1]["elements"][1]
            body, _ = _click(f, action["action_id"], action["value"])
            body["message"]["thread_ts"] = "1.0"
            await f._on_question(body, action)
            assert q.answer_free_text("PBE")
        result = await asyncio.wait_for(task, 2)
        answer = result.updated_input["answers"][QUESTIONS[count - 1]["question"]]
        assert str(path) in answer and "see plot" in answer
        assert s.queue.empty()
    asyncio.run(run())


def test_locked_message_forks_once(tmp_path: Path, monkeypatch):
    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER", session_id="existing")
        mirror = Mock()
        mirror.lock_text.return_value = "Terminal owns this session."
        f.hub.mirror = mirror
        submit = Mock()
        monkeypatch.setattr(ThreadSession, "submit", submit)
        await f._prompt("D1", "4.0", "1.0", "UOWNER", "please continue")
        action = f.hub.web.calls[-1][1]["blocks"][-1]["elements"][0]
        body, _ = _click(f, action["action_id"], action["value"])
        body["message"]["thread_ts"] = "1.0"
        await f._on_fork_send(body, action)
        await f._on_fork_send(body, action)
        submit.assert_called_once_with("please continue", "UOWNER")
        fork, = [r for r in f.hub.store.records.values() if r.key != s.rec.key]
        assert fork.fork_from == "existing"
        assert any("slack.example" in t for t in _texts(f))
    asyncio.run(run())


def test_approval_card_matches_scope(tmp_path: Path):
    from claude_agent_sdk import ToolPermissionContext

    async def run():
        f = _front(tmp_path)
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        for command, eligible in [("make report", True), ("rm old", False)]:
            task = asyncio.create_task(f.hub.request_approval(s, "Bash", {"command": command},
                                                              ToolPermissionContext(), "test"))
            await asyncio.sleep(0)
            aid = next(iter(f.hub.approvals))
            actions = f.hub.web.calls[-1][1]["blocks"][-1]["elements"]
            assert actions[0]["text"]["text"] == "Run once"
            assert any(a["action_id"] == "sac_remember" for a in actions) == eligible
            assert actions[-1]["type"] == "overflow" and actions[-1]["confirm"]
            if not eligible:
                assert f.hub.resolve_approval(aid, "remember", "UOWNER") == "forbidden"
            choice = "remember" if eligible else "approve"
            assert f.hub.resolve_approval(aid, choice, "UOWNER") == "ok"
            assert await task == (choice, "UOWNER")
    asyncio.run(run())
