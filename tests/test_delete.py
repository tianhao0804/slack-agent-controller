"""Deleting a thread-root message ends the session (offline, fake Slack)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from slack_agent_controller.app import SlackFrontend
from tests.helpers import FakeWeb, make_config


def _frontend(tmp: Path) -> SlackFrontend:
    f = SlackFrontend(make_config(tmp, [tmp]))
    f.hub.web = FakeWeb()  # type: ignore[assignment]
    return f


async def _settle(f: SlackFrontend) -> None:
    for _ in range(20):
        if not f._tasks:
            return
        await asyncio.sleep(0.05)


def _deleted(ts: str, author: str = "UOWNER") -> dict:
    return {"type": "message", "subtype": "message_deleted", "channel": "D1", "channel_type": "im",
            "deleted_ts": ts, "previous_message": {"user": author, "ts": ts, "text": "hi"}}


def _tombstone(ts: str, author: str = "UOWNER") -> dict:
    return {"type": "message", "subtype": "message_changed", "channel": "D1", "channel_type": "im",
            "message": {"subtype": "tombstone", "ts": ts, "text": "This message was deleted."},
            "previous_message": {"user": author, "ts": ts, "text": "hi"}}


def test_delete_root_ends_session(tmp_path: Path):
    async def run():
        f = _frontend(tmp_path)
        f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        f.hub.create_session("D1", "2.0", tmp_path, "UOWNER")
        await f._on_message(_deleted("1.0"))
        await f._on_message(_tombstone("2.0"))
        await _settle(f)
        assert f.hub.store.get("D1:1.0") is None and "D1:1.0" not in f.hub.sessions
        assert f.hub.store.get("D1:2.0") is None
        posts = [kw["text"] for op, kw in f.hub.web.calls if op == "post"]  # type: ignore[attr-defined]
        assert sum("ended (root message deleted" in t for t in posts) == 2
        audit = (f.cfg.state_dir / "audit.jsonl").read_text()
        assert audit.count('"session-ended"') == 2
    asyncio.run(run())


def test_ignored_cases(tmp_path: Path):
    async def run():
        f = _frontend(tmp_path)
        f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        # ordinary edit of the root
        await f._on_message({"type": "message", "subtype": "message_changed", "channel": "D1",
                             "channel_type": "im", "message": {"ts": "1.0", "text": "edited"},
                             "previous_message": {"user": "UOWNER", "ts": "1.0"}})
        # deleting a reply inside the thread
        await f._on_message(_deleted("1.5"))
        # root authored by someone not on the allowlist
        await f._on_message(_deleted("1.0", author="UEVIL"))
        await _settle(f)
        assert f.hub.store.get("D1:1.0") is not None
    asyncio.run(run())
