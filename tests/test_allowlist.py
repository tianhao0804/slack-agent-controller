"""Commands you approved in Slack are remembered (exact text + directory); hard rules still win."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from unittest.mock import AsyncMock

from claude_agent_sdk import ToolPermissionContext

from slack_agent_controller.allowlist import Allowlist, make_key
from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


def _hub(tmp: Path, **over) -> Hub:
    return Hub(make_config(tmp, [tmp], **over), FakeWeb())  # type: ignore[arg-type]


def _hook_input(command: str, cwd: Path, tid: str = "t1", **extra) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command, **extra}, "cwd": str(cwd), "tool_use_id": tid}


def _decision(out: dict) -> str:
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "defer")


async def _approve(sess, command: str, cwd: Path, tid: str = "t1", choice: str = "approve") -> str:
    """Run one Bash call through hook -> approval card -> can_use_tool; returns the final behavior."""
    tin = {"command": command}
    out = await sess._pre_tool_hook(_hook_input(command, cwd, tid), tid, None)
    if _decision(out) == "allow":
        return "allowed-by-hook"
    sess.hub.request_approval = AsyncMock(return_value=(choice, "UOWNER"))
    res = await sess._can_use_tool("Bash", tin, ToolPermissionContext(tool_use_id=tid))
    return res.behavior


def test_approved_command_is_remembered_and_exact(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        cmd = "python3 train.py --epochs 3"
        assert await _approve(s, cmd, tmp_path) == "allow"
        assert not hub.allowlist.entries
        assert await _approve(s, cmd, tmp_path, choice="remember") == "allow"                    # first time: card, approved
        assert len(hub.allowlist.entries) == 1
        assert await _approve(s, cmd, tmp_path, "t2") == "allowed-by-hook"    # now silent
        assert await _approve(s, cmd + " ", tmp_path, "t3") == "allowed-by-hook"  # surrounding whitespace only
        for other in (cmd + " --lr 1", "python3 train.py --epochs 4", "python3 train.py", "echo hi; " + cmd):
            s.hub.request_approval = AsyncMock(return_value=("deny", "UOWNER"))
            assert _decision(await s._pre_tool_hook(_hook_input(other, tmp_path), "t9", None)) == "ask", other
        audit = (hub.cfg.state_dir / "audit.jsonl").read_text()
        assert '"allowlist-add"' in audit and '"by": "allowlist"' in audit

    asyncio.run(run())


def test_denied_and_unlearnable_commands_are_not_remembered(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        assert await _approve(s, "python3 a.py", tmp_path, "t1", "deny") == "deny"
        assert not hub.allowlist.entries
        for cmd in (
            "curl https://example.invalid/x",        # network: flagged
            "rm -rf build",                          # destructive
            "/bin/rm old.txt",                       # destructive via absolute path
            "find . -name '*.tmp' -delete",          # destructive
            "git push origin main",                  # destructive
            "ls *.txt",                              # wildcard
            "echo $HOME",                            # variable
            "echo `date`",                           # backticks
            "cat .env",                              # protected file name
            "x" * 5000,                              # too long
        ):
            assert await _approve(s, cmd, tmp_path, "t2") == "allow", cmd     # approved once ...
            assert not hub.allowlist.entries, cmd                            # ... but not remembered
            out = await s._pre_tool_hook(_hook_input(cmd, tmp_path, "t3"), "t3", None)
            assert _decision(out) != "allow", cmd                            # and asks again
        # sandbox escape requests are never matched
        assert await _approve(s, "make all", tmp_path, "t4") == "allow"
        out = await s._pre_tool_hook(_hook_input("make all", tmp_path, "t5", dangerouslyDisableSandbox=True), "t5", None)
        assert _decision(out) == "ask"

    asyncio.run(run())


def test_directory_is_part_of_the_key(tmp_path: Path):
    async def run():
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        assert await _approve(s, "python3 run.py", tmp_path / "a", choice="remember") == "allow"
        assert await _approve(s, "python3 run.py", tmp_path / "a", "t2") == "allowed-by-hook"
        out = await s._pre_tool_hook(_hook_input("python3 run.py", tmp_path / "b", "t3"), "t3", None)
        assert _decision(out) == "ask", "same text in another directory asks again"

    asyncio.run(run())


def test_hard_denials_beat_the_allowlist(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        bad = "echo $AWS_BEARER_TOKEN_BEDROCK"
        key = make_key(bad, tmp_path)
        from slack_agent_controller.allowlist import Entry
        hub.allowlist.entries[key] = Entry(key, bad, str(tmp_path), 0.0, "UOWNER", "s")   # e.g. hand-edited file
        out = await s._pre_tool_hook(_hook_input(bad, tmp_path), "t1", None)
        assert _decision(out) == "deny"

    asyncio.run(run())


def test_builtin_commands_and_approve_all_interplay(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        for cmd in ("nvidia-smi", "git status", "git diff --stat", "git log --oneline -n 20"):
            out = await s._pre_tool_hook(_hook_input(cmd, tmp_path), "t1", None)
            assert _decision(out) == "allow", cmd
        for cmd in ("git status; rm x", "git log -p", "nvidia-smi -f out.txt", "git diff", "git branch -D x"):
            out = await s._pre_tool_hook(_hook_input(cmd, tmp_path), "t1", None)
            assert _decision(out) == "ask", cmd
        # "Approve all" still does not cover flagged commands (review R2) and does not add entries by itself
        s.allow_all.add("Bash")
        s.hub.request_approval = AsyncMock(return_value=("deny", "UOWNER"))
        res = await s._can_use_tool("Bash", {"command": "curl https://example.invalid"}, ToolPermissionContext())
        assert res.behavior == "deny"

    asyncio.run(run())


def test_learning_can_be_turned_off(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path, learn_approved_commands=False)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        assert await _approve(s, "python3 a.py", tmp_path) == "allow"
        assert not hub.allowlist.entries
        assert await _approve(s, "python3 a.py", tmp_path, "t2") == "allow"     # asked again (not by-hook)
        assert hub.allowlist.card_note({"command": "python3 a.py"}, tmp_path) == ""

    asyncio.run(run())


def test_persistence_permissions_and_corruption(tmp_path: Path):
    hub = _hub(tmp_path)
    al = hub.allowlist
    ent, status = al.learn({"command": "python3 a.py"}, tmp_path, "UOWNER", "sess")
    assert ent is not None and status == "added"
    assert al.learn({"command": "python3 a.py"}, tmp_path, "UOWNER", "sess")[1] == "exists"
    assert stat.S_IMODE(al.path.stat().st_mode) == 0o600
    again = Allowlist(al.path, hub.cfg)
    assert again.allowed({"command": "python3 a.py"}, tmp_path)
    # one malformed entry is skipped, the rest load
    data = json.loads(al.path.read_text())
    data["entries"].append({"bogus": 1})
    al.path.write_text(json.dumps(data))
    assert len(Allowlist(al.path, hub.cfg).entries) == 1
    # a corrupt file is moved aside and everything asks again (fails closed)
    al.path.write_text("{ not json")
    fresh = Allowlist(al.path, hub.cfg)
    assert not fresh.entries and list(tmp_path.glob("state/allowlist.json.corrupt-*"))
    assert fresh.allowed({"command": "python3 a.py"}, tmp_path) is None


def test_allowlist_command_list_remove_clear(tmp_path: Path):
    async def run():
        f = SlackFrontend(make_config(tmp_path, [tmp_path]))
        web = FakeWeb()
        f.hub.web = web  # type: ignore[assignment]
        al = f.hub.allowlist
        e1, _ = al.learn({"command": "python3 one.py"}, tmp_path, "UOWNER", "s")
        e2, _ = al.learn({"command": "python3 two.py\nsecond line"}, tmp_path, "UOWNER", "s")
        assert e1 and e2
        await f._cmd_allowlist("D1", "1.0", None, "UOWNER", "")
        text = web.texts()[-1]
        assert e1.id in text and e2.id in text and "python3 two.py…" in text and "nvidia-smi" in text
        await f._cmd_allowlist("D1", "1.1", None, "UOWNER", "remove zz")
        assert "at least 4" in web.texts()[-1] and len(al.entries) == 2
        await f._cmd_allowlist("D1", "1.2", None, "UOWNER", f"remove {e1.id}")
        assert "Removed" in web.texts()[-1] and len(al.entries) == 1
        assert al.allowed({"command": "python3 one.py"}, tmp_path) is None
        await f._cmd_allowlist("D1", "1.3", None, "UOWNER", "bogus")
        assert "Usage" in web.texts()[-1]
        await f._cmd_allowlist("D1", "1.4", None, "UOWNER", "clear")
        assert "Cleared 1" in web.texts()[-1] and not al.entries
        audit = (f.cfg.state_dir / "audit.jsonl").read_text()
        assert '"allowlist-remove"' in audit and '"allowlist-clear"' in audit

    asyncio.run(run())


def test_card_note_says_what_approving_does(tmp_path: Path):
    hub = _hub(tmp_path)
    al = hub.allowlist
    assert "saves this exact command" in al.card_note({"command": "python3 a.py"}, tmp_path)
    assert "always asks" in al.card_note({"command": "rm -rf x"}, tmp_path)
    assert al.card_note({"command": "nvidia-smi"}, tmp_path) == ""


def test_unparseable_command_is_never_matched(tmp_path: Path):
    hub = _hub(tmp_path)
    tin = {"command": "cat a\x00b"}
    assert hub.allowlist.blocked_reason(tin, tmp_path) is not None
    assert hub.allowlist.allowed(tin, tmp_path) is None
    assert hub.allowlist.learn(tin, tmp_path, "UOWNER", "s")[0] is None
