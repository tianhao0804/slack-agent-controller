"""Auto mode: the CLI's classifier judges everything except a short list (secrets,
permission files, environment dumps), which still needs a Slack tap. Ask mode is unchanged."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from slack_agent_controller import policy
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    claude = home / ".claude"
    (claude / "jobs" / "abcd1234" / "tmp").mkdir(parents=True)
    (claude / "projects" / "-x" / "memory").mkdir(parents=True)
    (claude / "settings.json").write_text("{}")
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("k")
    bot_cfg = home / ".config" / "slack-agent-controller"
    bot_cfg.mkdir(parents=True)
    (bot_cfg / "env").write_text("X=1")
    proj = home / "proj"
    proj.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    cfg = make_config(tmp_path, [home], sensitive=[claude, home / ".ssh", bot_cfg])
    return cfg, home, proj


def auto_w(cfg, cmd, cwd):
    return policy.auto_bash_warnings(cfg, cmd, cwd)


TMP = "~/.claude/jobs/abcd1234/tmp"


@pytest.mark.parametrize("cmd", [
    f"diff a.csv {TMP}/reg.csv",
    f"python3 - <<'EOF'\nopen('{TMP}/x.json','w').write('1')\nEOF",
    "cat ~/.claude/projects/-x/memory/MEMORY.md",
    "ls ~/.claude/jobs",
    "curl -sO https://example.org/paper.pdf",
    "wget https://example.org/data.tar.gz",
    "rsync -a results/ backup/",
    "find ~ -name '*.csv'",
])
def test_left_to_classifier_in_auto_mode(env, cmd):
    cfg, _home, proj = env
    assert auto_w(cfg, cmd, proj) == []


@pytest.mark.parametrize("cmd", [
    "cat ~/.claude/settings.json",
    "cat $HOME/.claude/settings.local.json",
    "cat ~/.claude/sett*",
    "cat ~/.claude/{settings,x}.json",
    f"cat {TMP}/../../../settings.json",
    "cd ~/.claude && cat settings.json",
    "D=~/.claude; cat $D/settings.json",
    "grep -r token ~/.claude",
    "cat ~/.ssh/id_ed25519",
    "cat ~/.config/slack-agent-controller/env",
    "cat proj/.env",
    "env",
    "printenv | sort",
    "python -c 'import os; print(os.environ)'",
])
def test_short_list_still_goes_to_slack(env, cmd):
    cfg, home, _proj = env
    assert auto_w(cfg, cmd, home), cmd


def test_symlink_in_job_tmp_to_a_secret_is_caught(env):
    cfg, home, proj = env
    link = home / ".claude" / "jobs" / "abcd1234" / "tmp" / "a"
    os.symlink(home / ".claude" / "settings.json", link)
    assert auto_w(cfg, f"cat {TMP}/a", proj)
    os.symlink(home / ".config" / "slack-agent-controller" / "env", link.with_name("b"))
    assert auto_w(cfg, f"cat {TMP}/b", proj)


def test_decide_non_bash_tools(env):
    cfg, home, proj = env
    tmpf = str(home / ".claude" / "jobs" / "abcd1234" / "tmp" / "x.csv")
    # auto: a non-secret file under ~/.claude goes to the classifier (ask, not forced)
    for tool, inp in [("Read", {"file_path": tmpf}), ("Write", {"file_path": tmpf, "content": "1"})]:
        r = policy.decide(cfg, tool, inp, proj, auto=True)
        assert r.decision == "ask" and not r.force_slack, (tool, r)
        assert policy.decide(cfg, tool, inp, proj).decision == "deny"  # ask mode unchanged
    # secrets and permission files stay blocked in both modes
    for f in (home / ".claude" / "settings.json", home / ".ssh" / "id_ed25519", home / ".config" / "slack-agent-controller" / "env",
              home / ".claude" / "settings.local.json"):
        for auto in (True, False):
            assert policy.decide(cfg, "Read", {"file_path": str(f)}, proj, auto=auto).decision == "deny"
            assert policy.decide(cfg, "Edit", {"file_path": str(f)}, proj, auto=auto).decision == "deny"
    # search scopes that hold a secret are still forced to Slack
    for scope in (home / ".claude", home):
        r = policy.decide(cfg, "Grep", {"pattern": "x", "path": str(scope)}, proj, auto=True)
        assert r.decision == "ask" and r.force_slack, scope
    r = policy.decide(cfg, "Grep", {"pattern": "x", "path": str(home / ".claude" / "jobs")}, proj, auto=True)
    assert r.decision == "ask" and not r.force_slack


def test_hard_denies_unchanged(env):
    cfg, _home, proj = env
    for cmd in ("echo $AWS_BEARER_TOKEN_BEDROCK", "cat /proc/self/environ"):
        assert policy.decide(cfg, "Bash", {"command": cmd}, proj, auto=True).decision == "deny"


def _hook(sess, cmd, cwd):
    inp = {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(cwd), "tool_use_id": "t1"}
    return asyncio.run(sess._pre_tool_hook(inp, "t1", None))


def test_session_routing_auto_vs_ask(env):
    cfg, _home, proj = env
    sess = Hub(cfg, FakeWeb()).create_session("D1", "1.0", proj, "UOWNER")  # type: ignore[arg-type]
    sess.rec.mode = "auto"
    assert _hook(sess, f"diff a {TMP}/b", proj) == {}  # classifier decides
    assert _hook(sess, "cat ~/.claude/settings.json", proj)["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert sess._force_slack("Bash", {"command": "env"}, proj)
    assert not sess._force_slack("Bash", {"command": "curl https://example.org"}, proj)
    sess.rec.mode = "ask"
    assert sess._force_slack("Bash", {"command": f"diff a {TMP}/b"}, proj)  # ask mode: full list
    assert sess._force_slack("Bash", {"command": "curl https://example.org"}, proj)


# ---------------------------------------------------------------------------
# Model / effort tag on every reply
# ---------------------------------------------------------------------------
def test_short_model_names():
    from slack_agent_controller.session import short_model
    assert short_model("us.anthropic.claude-opus-5-5") == "opus-5-5"
    assert short_model("global.anthropic.claude-sonnet-5-5-v1:0") == "sonnet-5-5"
    assert short_model("sonnet") == "sonnet"


def test_tail_model_effort_reads_last_assistant(tmp_path: Path):
    import json

    from slack_agent_controller.session import tail_model_effort
    f = tmp_path / "t.jsonl"
    rows = [{"type": "assistant", "requestedModel": "us.anthropic.claude-sonnet-5-5", "effort": "low",
             "message": {"model": "claude-sonnet-5-5"}},
            {"type": "user", "message": {"content": "x" * 5000}},
            {"type": "assistant", "requestedModel": "us.anthropic.claude-opus-5-5", "effort": "high",
             "message": {"model": "claude-opus-5-5"}},
            {"type": "system", "subtype": "turn_duration"}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert tail_model_effort(f) == ("us.anthropic.claude-opus-5-5", "high")
    assert tail_model_effort(f, max_bytes=60) == (None, None)  # tail with no assistant entry
    assert tail_model_effort(tmp_path / "missing.jsonl") == (None, None)


def test_model_tag_in_live_and_final_footer(env):
    from slack_agent_controller.session import TurnRenderer
    cfg, _home, proj = env
    sess = Hub(cfg, FakeWeb()).create_session("D1", "1.0", proj, "UOWNER")  # type: ignore[arg-type]
    assert "`default`" in sess.model_tag()
    sess.rec.effort = "medium"
    assert sess.model_tag().endswith("effort `medium`")
    sess.live_model, sess.live_effort = "us.anthropic.claude-opus-5-5", "high"
    assert sess.model_tag() == ":brain: `opus-5-5` · effort `high`"
    # the hook input reports the effort in effect for the main agent (not sub-agents)
    asyncio.run(sess._pre_tool_hook({"tool_name": "Read", "tool_input": {"file_path": str(proj / "a")},
                                     "cwd": str(proj), "effort": {"level": "max"}}, "t", None))
    assert sess.live_effort == "max"
    asyncio.run(sess._pre_tool_hook({"tool_name": "Read", "tool_input": {"file_path": str(proj / "a")},
                                     "cwd": str(proj), "effort": {"level": "low"}, "agent_id": "a1"}, "t", None))
    assert sess.live_effort == "max"
    r = TurnRenderer(sess.hub, "D1", "1.0", proj)
    r.tag = sess.model_tag
    r.commit_text("hello")
    assert r._compose(None).endswith("interrupt_  ·  :brain: `opus-5-5` · effort `max`")
    assert r._compose(":white_check_mark: done").endswith("done  ·  :brain: `opus-5-5` · effort `max`")
