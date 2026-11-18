"""Slack Home tab: overview content, access control, buttons."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ToolPermissionContext

import slack_agent_controller.app as app_mod
from slack_agent_controller import home
from slack_agent_controller.app import SlackFrontend
from tests.helpers import FakeWeb, make_config


def _front(tmp: Path, **over: Any) -> SlackFrontend:
    f = SlackFrontend(make_config(tmp, [tmp], **over))
    f.hub.web = FakeWeb()  # type: ignore[assignment]

    async def fake_run(*cmd: str, timeout: float = 15.0) -> str:
        return "0, 87, 30000, 46068\n1, 0, 10, 46068"

    f.hub._run = fake_run  # type: ignore[method-assign]
    return f


def _published(f: SlackFrontend) -> list[dict]:
    return [kw for op, kw in f.hub.web.calls if op == "views_publish"]  # type: ignore[attr-defined]


def _assert_slack_valid(view: dict) -> None:
    """Rules Slack enforced on the real views.publish: non-empty values, unique action ids."""
    ids: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("type") == "button":
                assert x.get("url") or x.get("value"), f"button without value: {x}"
                ids.append(x["action_id"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(view)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_home_shows_needs_you_active_gpus_recent_and_projects(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app_mod, "_titles", lambda ids: {i: "Phonon workflow" for i in ids})

    async def run():
        proj = tmp_path / "ml"
        proj.mkdir()
        f = _front(tmp_path, projects=(("ml", str(proj)),))
        busy = f.hub.create_session("D1", "1.0", proj, "UOWNER")
        f.hub.create_session("D1", "2.0", tmp_path, "UOWNER")
        busy._turn = object()  # type: ignore[assignment]
        # a pending approval and a pending question
        appr = asyncio.create_task(f.hub.request_approval(busy, "Bash", {"command": "make -j8"},
                                                          ToolPermissionContext(), "side effects"))
        ques = asyncio.create_task(f.hub.ask_questions(busy, {"questions": [
            {"question": "Which functional?", "header": "XC", "multiSelect": False,
             "options": [{"label": "PBE", "description": ""}, {"label": "r2SCAN", "description": ""}]}]}))
        for _ in range(50):
            if f.hub.approvals and f.hub.questions:
                break
            await asyncio.sleep(0.01)
        view = await home.build_home(f, "UOWNER")
        body = json.dumps(view)
        assert view["type"] == "home" and len(view["blocks"]) <= 100
        assert "Needs you* (2)" in body and "make -j8" in body and "Which functional?" in body
        assert "*Active* (1)" in body and "Phonon workflow" in body and "`ml`" in body
        assert "GPU0" in body and "87%" in body
        assert "Recent sessions" in body
        assert '"sac_home_proj_ml"' in body and '"sac_home_refresh"' in body
        _assert_slack_valid(view)
        opens = [b["accessory"] for b in view["blocks"] if b.get("accessory", {}).get("url")]
        assert len(opens) >= 4, "approval, question, active and recent rows link to Slack"
        for aid in list(f.hub.approvals):
            f.hub.resolve_approval(aid, "deny", "UOWNER")
        for q in list(f.hub.questions.values()):
            q.future.set_result(("skip", None))
        await asyncio.gather(appr, ques)
        busy._turn = None
        quiet = json.dumps(await home.build_home(f, "UOWNER"))
        assert "nothing right now" in quiet and "no session is running" in quiet
    asyncio.run(run())


def test_home_opened_event_and_access_control(tmp_path: Path):
    async def run():
        f = _front(tmp_path)
        await f._publish_home("UOWNER")
        await f._publish_home("UEVIL")
        pub = _published(f)
        assert pub[0]["user_id"] == "UOWNER" and "Agent Controller" in json.dumps(pub[0]["view"])
        assert pub[1]["user_id"] == "UEVIL" and "allowlisted" in json.dumps(pub[1]["view"])
        assert "GPU" not in json.dumps(pub[1]["view"])
    asyncio.run(run())


def test_home_new_session_buttons_open_in_the_dm(tmp_path: Path):
    async def run():
        proj = tmp_path / "web"
        proj.mkdir()
        f = _front(tmp_path, projects=(("web", str(proj)),))
        body = {"user": {"id": "UOWNER"}}
        await f._on_home_action(body, {"action_id": "sac_home_proj_soap", "value": "web"})
        recs = list(f.hub.store.records.values())
        assert len(recs) == 1 and recs[0].channel == "DUOWNER" and recs[0].cwd == str(proj)
        await f._on_home_action(body, {"action_id": "sac_home_new", "value": ""})
        assert len(f.hub.store.records) == 2
        assert len(_published(f)) == 2, "the Home tab is refreshed after each click"
        await f._on_home_action({"user": {"id": "UEVIL"}}, {"action_id": "sac_home_new", "value": ""})
        assert len(f.hub.store.records) == 2
        await f._on_home_action(body, {"action_id": "sac_home_proj_nope", "value": "nope"})
        assert len(f.hub.store.records) == 2
    asyncio.run(run())
