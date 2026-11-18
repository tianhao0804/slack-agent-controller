"""Build Slack-ready excerpts of an existing Claude Code session (user-triggered).

Only the user's prompts and Claude's text replies are kept; tool calls are
reduced to one-line summaries and tool outputs / thinking are dropped. All
text is passed through ``policy.redact``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import SDKSessionInfo, get_session_messages, list_sessions

from . import policy


@dataclass
class Exchange:
    prompt: str
    is_summary: bool = False  # the CLI's context summary after compaction, not a user prompt
    replies: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


def resolve_ids(prefixes: list[str]) -> tuple[list[SDKSessionInfo], list[str]]:
    """Map id prefixes (at least 6 chars) to sessions. Returns (found, errors)."""
    infos = list_sessions()
    found: list[SDKSessionInfo] = []
    errors: list[str] = []
    for pre in prefixes:
        if len(pre) < 6:
            errors.append(f"`{pre}`: give at least 6 characters of the session id")
            continue
        matches = [i for i in infos if i.session_id.startswith(pre)]
        if len(matches) == 1:
            found.append(matches[0])
        elif not matches:
            errors.append(f"`{pre}`: no such session on this node")
        else:
            errors.append(f"`{pre}`: ambiguous ({len(matches)} sessions match)")
    return found, errors


def _is_injected(text: str) -> bool:
    """Prompts the CLI inserts itself (slash-command echoes, reminders, ...)."""
    t = text.lstrip()
    return t.startswith("<") or t.startswith("Caveat:")


_SUMMARY_PREFIX = "This session is being continued from a previous conversation"


def exchanges(info: SDKSessionInfo) -> list[Exchange]:
    cwd = Path(info.cwd or ".")
    out: list[Exchange] = []
    # No ``directory``: the SDK would canonicalise symlinked cwds (~/proj ->
    # /data/proj) and miss transcripts stored under the original spelling.
    for m in get_session_messages(info.session_id):
        content: Any = (m.message or {}).get("content")
        if m.type == "user":
            texts = [content] if isinstance(content, str) else [
                b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"
            ]
            texts = [t for t in texts if t.strip() and not _is_injected(t)]
            if texts:  # tool results arrive as user messages without text: skipped
                joined = "\n".join(texts)
                out.append(Exchange(prompt=policy.redact(joined), is_summary=joined.lstrip().startswith(_SUMMARY_PREFIX)))
        elif m.type == "assistant" and out and isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and str(b.get("text", "")).strip():
                    out[-1].replies.append(policy.redact(str(b["text"])))
                elif b.get("type") == "tool_use":
                    line = policy.tool_line(str(b.get("name", "")), b.get("input") or {}, cwd)
                    out[-1].tools.append(policy.redact(line))
    return out


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def format_exchange(e: Exchange, index: int, total: int) -> str:
    """One Slack message (mrkdwn) for one exchange."""
    # Imported lazily to keep this module free of Slack details at import time.
    from .slackfmt import escape, head_text, inline_code, md_to_mrkdwn

    if e.is_summary:
        head = f"*:compression: Context summary* _(written by Claude Code when the conversation was compacted; {index}/{total})_"
        body = "\n".join("> " + ln for ln in escape(_clip(e.prompt, 1200)).splitlines())
    else:
        head = f"*:bust_in_silhouette: You* _({index}/{total})_"
        body = "\n".join("> " + ln for ln in escape(_clip(e.prompt, 1200)).splitlines())
    parts = [head, body]
    if e.replies:
        parts += ["*:robot_face: Claude*", head_text(md_to_mrkdwn("\n\n".join(e.replies)), 1900)]
    if e.tools:
        shown = "\n".join(":hammer_and_wrench: " + inline_code(_clip(t, 100)) for t in e.tools[-4:])
        more = f"_{len(e.tools)} tool calls (outputs not imported)" + (", last 4:_" if len(e.tools) > 4 else ":_")
        parts += [more, shown]
    return "\n".join(parts)


def card_text(info: SDKSessionInfo, n_total: int, n_shown: int, last_active: str, node: str = "this node") -> str:
    from .slackfmt import escape, inline_code

    title = escape(_clip(info.custom_title or info.summary or "(untitled session)", 120))
    meta = [inline_code(info.cwd or "?")]
    if info.git_branch:
        meta.append(f"branch {inline_code(info.git_branch)}")
    meta += [f"last active {last_active}", f"{n_total} exchanges in current context"]
    return (
        f":card_index_dividers: *{title}*\n" + " · ".join(meta) + f"\nsession `{info.session_id}`\n"
        f"_Imported from {node}: last {n_shown} exchanges (your prompts and Claude's replies; tool outputs omitted, "
        "secrets redacted). Reply in this thread to continue this session._\n"
        ":warning: If this session is open in a terminal right now, don't reply here at the same time — send "
        "`!fork` in this thread to branch it into a new one instead."
    )
