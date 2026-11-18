"""Offline unit tests: permission policy and Slack formatting."""

from __future__ import annotations

from pathlib import Path

import pytest

from slack_agent_controller import policy
from slack_agent_controller.slackfmt import md_to_mrkdwn, slack_to_plain, tail_text
from tests.helpers import make_config


@pytest.fixture
def env(tmp_path: Path):
    work = tmp_path / "proj"
    work.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    cfg = make_config(tmp_path, roots=[tmp_path], sensitive=[secret])
    return cfg, work, secret


def d(cfg, tool, inp, cwd):
    return policy.decide(cfg, tool, inp, cwd).decision


def test_read_inside_root_allowed(env):
    cfg, work, _ = env
    assert d(cfg, "Read", {"file_path": str(work / "a.py")}, work) == "allow"
    assert d(cfg, "Read", {"file_path": "rel/a.py"}, work) == "allow"


def test_read_outside_root_asks(env):
    cfg, work, _ = env
    assert d(cfg, "Read", {"file_path": "/etc/hostname"}, work) == "ask"


def test_cli_task_output_dir_readable(tmp_path):
    import dataclasses
    ro = tmp_path / "claude-tasks"
    work = tmp_path / "work"
    ro.mkdir()
    work.mkdir()
    cfg = dataclasses.replace(make_config(tmp_path, roots=[work]), read_only_roots=(ro.resolve(),))
    assert d(cfg, "Read", {"file_path": str(ro / "t.output")}, work) == "allow"
    assert d(cfg, "Write", {"file_path": str(ro / "t.output"), "content": ""}, work) == "deny"


def test_sensitive_paths_denied(env):
    cfg, work, secret = env
    assert d(cfg, "Read", {"file_path": str(secret / "k")}, work) == "deny"
    assert d(cfg, "Edit", {"file_path": str(secret / "k")}, work) == "deny"
    assert d(cfg, "Read", {"file_path": str(work / ".env")}, work) == "deny"
    assert d(cfg, "Read", {"file_path": str(work / "id_rsa")}, work) == "deny"
    # symlink into a protected dir is resolved
    (work / "link").symlink_to(secret)
    assert d(cfg, "Read", {"file_path": str(work / "link" / "k")}, work) == "deny"


def test_search_scope_containing_secrets_asks(env):
    cfg, work, _ = env
    root = work.parent
    assert d(cfg, "Grep", {"pattern": "x"}, root) == "ask"  # default path = cwd contains secret
    assert d(cfg, "Grep", {"pattern": "x"}, work) == "allow"
    assert d(cfg, "Glob", {"pattern": str(root) + "/**/*.py"}, work) == "ask"


def test_writes(env):
    cfg, work, _ = env
    assert d(cfg, "Write", {"file_path": str(work / "o.txt"), "content": ""}, work) == "ask"
    assert d(cfg, "Write", {"file_path": "/tmp/x.txt", "content": ""}, work) == "deny"


def test_bash(env):
    cfg, work, _ = env
    assert d(cfg, "Bash", {"command": "python3 train.py"}, work) == "ask"
    assert d(cfg, "Bash", {"command": "echo $AWS_BEARER_TOKEN_BEDROCK"}, work) == "deny"
    assert d(cfg, "Bash", {"command": "cat /proc/self/environ"}, work) == "deny"
    assert policy.bash_warnings(cfg, "env | sort")
    assert policy.bash_warnings(cfg, "ls; printenv")
    assert not policy.bash_warnings(cfg, "ls -la && nvidia-smi")


def test_unknown_tool_asks(env):
    cfg, work, _ = env
    assert d(cfg, "WebFetch", {"url": "https://x"}, work) == "ask"
    assert d(cfg, "mcp__foo__bar", {}, work) == "ask"


def test_redact():
    assert "abc123" not in policy.redact("TOKEN=abc123 password: hunter2")
    assert "hunter2" not in policy.redact("TOKEN=abc123 password: hunter2")


def test_mrkdwn():
    assert md_to_mrkdwn("**bold** and *it*") == "*bold* and _it_"
    assert md_to_mrkdwn("## Title") == "*Title*"
    assert md_to_mrkdwn("[a](https://x.y)") == "<https://x.y|a>"
    assert md_to_mrkdwn("```python\nx = 1 < 2\n```") == "```x = 1 &lt; 2```"
    assert md_to_mrkdwn("- item") == "• item"
    assert md_to_mrkdwn("`**not bold**`") == "`**not bold**`"
    # unterminated fence while streaming
    assert md_to_mrkdwn("hi\n```bash\nls").endswith("```")


def test_slack_to_plain():
    assert slack_to_plain("open <http://example.com|example.com> &amp; &lt;x&gt;") == "open example.com & <x>"
    assert slack_to_plain("<https://a.b/c>") == "https://a.b/c"


def test_tail_text_balances_fences():
    s = "```" + "x\n" * 3000 + "```"
    t = tail_text(s, 500)
    assert len(t) < 600 and t.count("```") % 2 == 0


def test_bash_is_never_auto_allowed_by_policy(env):
    """The program-name rule is gone (review R1): policy never allows Bash by itself."""
    cfg, work, secret = env
    for c in ("ls -la", "tail -n 50 run.log", "nvidia-smi", "git status", "du -sh .", "grep -rn TODO .",
              "sort -o out.txt in.txt", "sort -o../outside.txt in.txt", "rg --pre=./x.sh foo ."):
        assert d(cfg, "Bash", {"command": c}, work) == "ask", c
    for c in ("echo $AWS_BEARER_TOKEN_BEDROCK", "cat /proc/self/environ", "cat /proc/1/environ"):
        assert d(cfg, "Bash", {"command": c}, work) == "deny", c


def test_redact_known_token_formats():
    fakes = ["ABSK" + "Q" * 40, "bedrock-api-key-" + "Z" * 30, "xoxb-" + "1" * 12 + "-abc", "xapp-1-" + "A" * 20,
             "AKIA" + "B" * 16, "ghp_" + "c" * 30, "sk-" + "d" * 30,
             "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n-----END " + "OPENSSH PRIVATE KEY-----"]
    for f in fakes:
        out = policy.redact(f"value: {f} end")
        assert f not in out and "<redacted" in out, f
    assert policy.redact("plain text stays") == "plain text stays"
