"""Test helpers: build a Config without an env file, and a fake Slack client."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from slack_agent_controller.config import _DEFAULT_SENSITIVE_GLOBS, Config


def make_config(tmp: Path, roots: list[Path], sensitive: list[Path] | None = None, **over: Any) -> Config:
    state = tmp / "state"
    state.mkdir(exist_ok=True)
    kw: dict[str, Any] = dict(
        slack_bot_token="xoxb-test",
        slack_app_token="xapp-test",
        allowed_users=frozenset({"UOWNER"}),
        allowed_channels=frozenset(),
        cwd_roots=tuple(r.resolve() for r in roots),
        default_cwd=roots[0].resolve(),
        sensitive_paths=tuple(p.resolve() for p in (sensitive or [])) + (state.resolve(),),
        sensitive_globs=tuple(_DEFAULT_SENSITIVE_GLOBS),
        auto_allow_tools=frozenset(),
        disallowed_tools=(),
        approval_timeout_s=1800.0,
        state_dir=state,
        env_file=tmp / "env",
        model=None,
        max_budget_usd=2.0,
        max_turns=None,
        notify_after_s=60.0,
        idle_disconnect_s=7200.0,
        update_interval_s=1.0,
        cli_path=None,
        scrub_subprocess_env=False,
        load_claude_md=False,
        squeue_path=None,
        upload_max_mb=50.0,
        token_expires=None,
    )
    kw.update(over)
    return Config(**kw)


class FakeWeb:
    """Records Slack Web API calls; returns increasing message timestamps."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._ts = itertools.count(1000)
        self.messages: dict[str, dict[str, Any]] = {}

    async def chat_postMessage(self, **kw: Any) -> dict[str, Any]:  # noqa: N802
        ts = f"{next(self._ts)}.000"
        self.calls.append(("post", kw))
        self.messages[ts] = dict(kw)
        return {"ts": ts, "ok": True}

    async def chat_update(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("update", kw))
        self.messages.setdefault(kw["ts"], {}).update(kw)
        return {"ok": True}

    async def files_upload_v2(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("upload", kw))
        return {"ok": True}

    async def views_publish(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("views_publish", kw))
        return {"ok": True}

    async def conversations_open(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("conversations_open", kw))
        return {"ok": True, "channel": {"id": "D" + str(kw.get("users", "X"))}}

    async def chat_getPermalink(self, **kw: Any) -> dict[str, Any]:  # noqa: N802
        return {"permalink": "https://slack.example/p"}

    def texts(self) -> list[str]:
        return [m.get("text", "") for m in self.messages.values()]


def is_reply(kw: dict[str, Any]) -> bool:
    """A chat.update/post of a reply message (as opposed to an approval or question card)."""
    blocks = kw.get("blocks") or []
    return not blocks or str(blocks[0].get("block_id", "")).startswith("sac_reply")


def is_card(kw: dict[str, Any]) -> bool:
    """An approval card (its actions block has a sac_appr_ id)."""
    return any(str(b.get("block_id", "")).startswith("sac_appr_") for b in kw.get("blocks") or [])
