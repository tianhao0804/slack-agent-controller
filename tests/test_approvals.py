"""Approval precedence (review R2): "Approve all" never covers calls that must always ask."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from claude_agent_sdk import ToolPermissionContext

from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


def _session(tmp: Path):
    hub = Hub(make_config(tmp, [tmp]), FakeWeb())  # type: ignore[arg-type]
    return hub.create_session("D1", "1.0", tmp, "UOWNER")


def test_approve_all_does_not_cover_flagged_commands(tmp_path: Path):
    async def run():
        s = _session(tmp_path)
        s.allow_all.add("Bash")
        s.hub.request_approval = AsyncMock(return_value=("deny", "UOWNER"))
        flagged = {"command": "curl https://example.invalid"}
        assert s._force_slack("Bash", flagged)
        res = await s._can_use_tool("Bash", flagged, ToolPermissionContext())
        assert res.behavior == "deny"
        s.hub.request_approval.assert_awaited_once()

    asyncio.run(run())


def test_approve_all_still_covers_ordinary_commands(tmp_path: Path):
    async def run():
        s = _session(tmp_path)
        s.allow_all.add("Bash")
        s.hub.request_approval = AsyncMock(side_effect=AssertionError("should not ask"))
        res = await s._can_use_tool("Bash", {"command": "make -j4"}, ToolPermissionContext())
        assert res.behavior == "allow"

    asyncio.run(run())
