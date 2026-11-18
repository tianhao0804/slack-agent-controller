"""Env-file parsing and config validation (uses the real .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from slack_agent_controller.config import ConfigError, load_config, load_env_file

EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
KEYS = ["STATE_DIR", "CLAUDE_CLI_PATH", "SQUEUE_PATH", "EXTRA_SENSITIVE_PATHS", "EXTRA_SENSITIVE_GLOBS",
        "AUTO_ALLOW_EXTRA_TOOLS", "MAX_TURNS_PER_PROMPT", "ALLOWED_CHANNEL_IDS"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for k in list(os.environ):
        if k.startswith(("SLACK_", "AWS_", "ANTHROPIC_", "CLAUDE_", "SAC_")) or k in KEYS:
            monkeypatch.delenv(k, raising=False)
    for k in KEYS + ["CWD_ROOTS", "DEFAULT_CWD", "APPROVAL_TIMEOUT_SEC", "BEDROCK_TOKEN_EXPIRES"]:
        monkeypatch.delenv(k, raising=False)
    yield


def _write(tmp: Path, text: str) -> Path:
    p = tmp / "env"
    p.write_text(text)
    p.chmod(0o600)
    return p


def _example(tmp: Path, **override: str) -> Path:
    lines = []
    for line in EXAMPLE.read_text().splitlines():
        k = line.split("=", 1)[0]
        if k in override:
            line = f"{k}={override.pop(k)}"
        lines.append(line)
    lines += [f"{k}={v}" for k, v in override.items()]
    return _write(tmp, "\n".join(lines) + "\n")


def test_empty_values_with_inline_comments(tmp_path: Path):
    load_env_file(_write(tmp_path, "A=   # note\nB=value   # note\nC='x # not a comment'\nD=a#b\n"))
    assert os.environ["A"] == ""
    assert os.environ["B"] == "value"
    assert os.environ["C"] == "x # not a comment"
    assert os.environ["D"] == "a#b"


def test_example_file_parses_to_defaults(tmp_path: Path):
    home = str(Path.home())
    env = _example(tmp_path, SLACK_BOT_TOKEN="xoxb-x", SLACK_APP_TOKEN="xapp-x", AWS_BEARER_TOKEN_BEDROCK="t",
                   CWD_ROOTS=str(tmp_path), DEFAULT_CWD=str(tmp_path))
    cfg = load_config(env)
    assert cfg.cli_path is None
    assert cfg.state_dir == Path(home, ".local/state/slack-agent-controller").resolve()
    assert cfg.auto_allow_tools == frozenset()
    assert cfg.max_turns is None
    assert not any("#" in str(p) for p in cfg.sensitive_paths)
    assert not any("#" in g for g in cfg.sensitive_globs)
    assert "SLACK_BOT_TOKEN" not in os.environ


@pytest.mark.parametrize("tools", ["Bash", "WebSearch,Write", "Edit"])
def test_auto_allow_rejects_side_effect_tools(tmp_path: Path, tools: str):
    env = _example(tmp_path, SLACK_BOT_TOKEN="xoxb-x", SLACK_APP_TOKEN="xapp-x", AWS_BEARER_TOKEN_BEDROCK="t",
                   CWD_ROOTS=str(tmp_path), DEFAULT_CWD=str(tmp_path), STATE_DIR=str(tmp_path / "s"),
                   AUTO_ALLOW_EXTRA_TOOLS=tools)
    with pytest.raises(ConfigError, match="AUTO_ALLOW_EXTRA_TOOLS"):
        load_config(env)


def test_bad_cli_path_rejected(tmp_path: Path):
    env = _example(tmp_path, SLACK_BOT_TOKEN="xoxb-x", SLACK_APP_TOKEN="xapp-x", AWS_BEARER_TOKEN_BEDROCK="t",
                   CWD_ROOTS=str(tmp_path), DEFAULT_CWD=str(tmp_path), STATE_DIR=str(tmp_path / "s"),
                   CLAUDE_CLI_PATH="/nonexistent/claude")
    with pytest.raises(ConfigError, match="CLAUDE_CLI_PATH"):
        load_config(env)


def test_api_key_rejected(tmp_path: Path):
    env = _example(tmp_path, SLACK_BOT_TOKEN="xoxb-x", SLACK_APP_TOKEN="xapp-x", AWS_BEARER_TOKEN_BEDROCK="t",
                   CWD_ROOTS=str(tmp_path), DEFAULT_CWD=str(tmp_path), STATE_DIR=str(tmp_path / "s"),
                   ANTHROPIC_API_KEY="sk-x")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_config(env)


# -- providers ------------------------------------------------------------------------
def _provider_env(monkeypatch, **env: str) -> None:
    for k in ("PROVIDER", "CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AWS_REGION",
              "AWS_DEFAULT_REGION", "AWS_BEARER_TOKEN_BEDROCK", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
              "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_provider_bedrock(monkeypatch):
    import os

    from slack_agent_controller.config import check_provider
    _provider_env(monkeypatch, PROVIDER="bedrock", AWS_REGION="us-east-1", AWS_PROFILE="lab")
    assert check_provider() == "bedrock" and os.environ["CLAUDE_CODE_USE_BEDROCK"] == "1"
    _provider_env(monkeypatch, PROVIDER="bedrock", AWS_PROFILE="lab")
    with pytest.raises(ConfigError, match="AWS_REGION"):
        check_provider()
    _provider_env(monkeypatch, PROVIDER="bedrock", AWS_REGION="us-east-1", AWS_PROFILE="lab",
                  ANTHROPIC_API_KEY="sk-" + "ant-x")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is set but PROVIDER=bedrock"):
        check_provider()


def test_provider_anthropic(monkeypatch):
    from slack_agent_controller.config import check_provider
    key = "sk-" + "ant-" + "api03-" + "x" * 30
    _provider_env(monkeypatch, PROVIDER="anthropic", ANTHROPIC_API_KEY=key)
    assert check_provider() == "anthropic"
    _provider_env(monkeypatch, ANTHROPIC_API_KEY=key)                      # inferred
    assert check_provider() == "anthropic"
    _provider_env(monkeypatch, PROVIDER="anthropic", ANTHROPIC_API_KEY=key, CLAUDE_CODE_USE_BEDROCK="1")
    with pytest.raises(ConfigError, match="CLAUDE_CODE_USE_BEDROCK is set"):
        check_provider()
    _provider_env(monkeypatch, PROVIDER="anthropic", ANTHROPIC_API_KEY="nope")
    with pytest.raises(ConfigError, match="sk-ant-"):
        check_provider()


def test_provider_rejects_claude_ai_login_and_unknown(monkeypatch):
    from slack_agent_controller.config import check_provider
    _provider_env(monkeypatch, PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-" + "ant-" + "x" * 30,
                  CLAUDE_CODE_OAUTH_TOKEN="t")
    with pytest.raises(ConfigError, match="claude.ai login"):
        check_provider()
    _provider_env(monkeypatch, PROVIDER="openai")
    with pytest.raises(ConfigError, match="PROVIDER must be one of"):
        check_provider()
    _provider_env(monkeypatch)
    with pytest.raises(ConfigError, match="Set PROVIDER"):
        check_provider()


def test_session_env_follows_the_provider(tmp_path):
    import dataclasses

    from slack_agent_controller.hub import Hub
    from tests.helpers import FakeWeb, make_config
    cfg = make_config(tmp_path, [tmp_path])
    s = Hub(cfg, FakeWeb()).create_session("D1", "1.0", tmp_path, "U")  # type: ignore[arg-type]
    assert s._options().env.get("CLAUDE_CODE_USE_BEDROCK") == "1"
    cfg2 = dataclasses.replace(cfg, provider="anthropic")
    s2 = Hub(cfg2, FakeWeb()).create_session("D1", "2.0", tmp_path, "U")  # type: ignore[arg-type]
    assert "CLAUDE_CODE_USE_BEDROCK" not in s2._options().env
