"""Only the session's own project auto-memory directory is opened inside ~/.claude."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from slack_agent_controller import policy
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    proj = home / "my_proj"
    proj.mkdir()
    cfg = make_config(tmp_path, [home], sensitive=[home / ".claude", home / ".ssh"])
    mem = policy.project_memory_dir(proj)
    mem.mkdir(parents=True)
    return cfg, home, proj, mem


def d(cfg, tool, inp, cwd, mem):  # noqa: ANN001
    return policy.decide(cfg, tool, inp, cwd, mem).decision


def test_memory_dir_name_matches_claude_code(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/u"))
    assert policy.project_memory_dir("/home/u/my_proj") == Path("/home/u/.claude/projects/-home-u-my-proj/memory")
    assert policy.project_memory_dir("/data/a.b/x_y") == Path("/home/u/.claude/projects/-data-a-b-x-y/memory")


def test_own_memory_is_open_everything_else_in_claude_stays_closed(env):
    cfg, home, proj, mem = env
    note = str(mem / "notes.md")
    for tool, inp in [("Read", {"file_path": note}), ("Write", {"file_path": note, "content": "x"}),
                      ("Edit", {"file_path": str(mem / "MEMORY.md"), "old_string": "a", "new_string": "b"}),
                      ("Glob", {"pattern": "*.md", "path": str(mem)}), ("Grep", {"pattern": "x", "path": str(mem)})]:
        assert d(cfg, tool, inp, proj, mem) == "allow", tool
    other = policy.project_memory_dir(home / "other")
    other.mkdir(parents=True)
    closed = [
        ("Read", {"file_path": str(other / "notes.md")}),                          # another project's memory
        ("Read", {"file_path": str(mem.parent / "abc.jsonl")}),                     # transcripts next to it
        ("Read", {"file_path": str(home / ".claude" / "settings.json")}),
        ("Read", {"file_path": str(mem / ".." / ".." / ".." / "settings.json")}),   # .. out of memory/
        ("Grep", {"pattern": "x", "path": str(mem.parent)}),
        ("Read", {"file_path": str(home / ".ssh" / "id_rsa")}),
    ]
    for tool, inp in closed:
        assert d(cfg, tool, inp, proj, mem) == "deny", inp
    assert d(cfg, "Read", {"file_path": note}, proj, None) == "deny", "no memory dir given -> closed"


def test_bash_still_cannot_touch_claude_dir(env):
    cfg, home, proj, mem = env
    res = policy.decide(cfg, "Bash", {"command": f"cat {mem}/notes.md"}, proj, mem)
    assert res.decision == "ask"
    assert policy.bash_warnings(cfg, f"cat {mem}/notes.md", proj), "flagged: must always ask in Slack"


def test_symlinked_memory_dir_does_not_open_its_target(env):
    cfg, home, proj, mem = env
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("k")
    evil_proj = home / "evil"
    evil_proj.mkdir()
    evil_mem = policy.project_memory_dir(evil_proj)
    evil_mem.parent.mkdir(parents=True)
    os.symlink(home / ".ssh", evil_mem)
    assert d(cfg, "Read", {"file_path": str(evil_mem / "id_rsa")}, evil_proj, evil_mem) == "deny"


def test_session_pins_memory_dir_and_uses_it(env):
    cfg, home, proj, mem = env

    async def run():
        hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
        s = hub.create_session("D1", "1.0", proj, "UOWNER")
        assert s.memory_dir == mem
        assert json.loads(s._options().settings or "{}") == {"autoMemoryDirectory": str(mem)}
        assert str(mem) in s._system_append()
        out = await s._pre_tool_hook({"tool_name": "Write", "tool_input": {"file_path": str(mem / "n.md"),
                                      "content": "x"}, "cwd": str(proj)}, "t1", None)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    asyncio.run(run())
