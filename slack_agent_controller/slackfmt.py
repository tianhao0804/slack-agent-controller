"""Markdown -> Slack mrkdwn conversion and Slack text helpers."""

from __future__ import annotations

import html
import re

# Conservative limit for a single message body (Slack truncates long texts
# and section blocks cap at 3000 chars).
MSG_LIMIT = 3800
BLOCK_TEXT_LIMIT = 2900

_FENCE = re.compile(r"```[^\n`]*\n?(.*?)```", re.S)
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\*\w])")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)
_BULLET = re.compile(r"^(\s*)[-*+]\s+", re.M)
_STRIKE = re.compile(r"~~(.+?)~~")
_INLINE_CODE = re.compile(r"`[^`\n]+`")


def escape(s: str) -> str:
    """Escape the three characters Slack treats specially."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _convert_prose(s: str) -> str:
    # Protect inline code spans from formatting rewrites.
    codes: list[str] = []

    def stash(m: re.Match[str]) -> str:
        codes.append(m.group(0))
        return f"\x00{len(codes) - 1}\x00"

    s = _INLINE_CODE.sub(stash, s)
    s = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", s)
    s = _HEADING.sub(lambda m: f"\x01{m.group(1)}\x01", s)
    s = _BOLD.sub(lambda m: f"\x01{m.group(1) or m.group(2)}\x01", s)
    s = _ITALIC.sub(lambda m: f"_{m.group(1)}_", s)
    s = _BULLET.sub(lambda m: f"{m.group(1)}• ", s)
    s = _STRIKE.sub(lambda m: f"~{m.group(1)}~", s)
    s = s.replace("\x01", "*")
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)


def md_to_mrkdwn(md: str) -> str:
    """Best-effort conversion of Claude's Markdown into Slack mrkdwn."""
    md = escape(md)
    out: list[str] = []
    pos = 0
    for m in _FENCE.finditer(md):
        out.append(_convert_prose(md[pos : m.start()]))
        out.append("```" + m.group(1).rstrip("\n") + "```")
        pos = m.end()
    tail = md[pos:]
    # An unterminated fence while streaming: render the rest as code.
    if "```" in tail:
        i = tail.index("```")
        out.append(_convert_prose(tail[:i]))
        body = tail[i + 3 :].split("\n", 1)
        out.append("```" + (body[1] if len(body) > 1 else "") + "```")
    else:
        out.append(_convert_prose(tail))
    return "".join(out)


def tail_text(s: str, limit: int = MSG_LIMIT) -> str:
    """Keep the last ``limit`` chars, keeping code fences balanced."""
    if len(s) <= limit:
        return s
    cut = s[-limit:]
    nl = cut.find("\n")
    if 0 <= nl < 200:
        cut = cut[nl + 1 :]
    if cut.count("```") % 2 == 1:
        cut = "```" + cut
    return "…(earlier output truncated)…\n" + cut


def head_text(s: str, limit: int = MSG_LIMIT) -> str:
    if len(s) <= limit:
        return s
    cut = s[:limit]
    if cut.count("```") % 2 == 1:
        cut += "```"
    return cut + "\n…(truncated)…"


_TRUNC_NOTE = "\n…(truncated; full text attached)…"


def fit_code_block(s: str, limit: int = BLOCK_TEXT_LIMIT) -> tuple[str, bool]:
    """A ``` block whose *escaped* length stays within ``limit`` (escaping can
    multiply the length, e.g. ``&`` -> ``&amp;``). Returns (block, truncated)."""
    esc = escape(s.replace("```", "ʼʼʼ"))
    room = limit - 6  # the two fences
    if len(esc) <= room:
        return f"```{esc}```", False
    cut = esc[: room - len(_TRUNC_NOTE)]
    amp = cut.rfind("&", max(0, len(cut) - 5))
    if amp != -1 and ";" not in cut[amp:]:
        cut = cut[:amp]  # do not split an entity such as &amp;
    return f"```{cut}{_TRUNC_NOTE}```", True


def code_block(s: str, limit: int = BLOCK_TEXT_LIMIT) -> str:
    return fit_code_block(s, limit)[0]


def text_blocks(text: str, limit: int = BLOCK_TEXT_LIMIT, block_id: str | None = None) -> list[dict]:
    """mrkdwn text as section blocks of at most ``limit`` chars each, split on
    line boundaries; a code fence cut by a split is closed and reopened."""
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single huge line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        cand = f"{cur}\n{line}" if cur else line
        if len(cand) > limit - 4:
            chunks.append(cur)
            cur = line
        else:
            cur = cand
    if cur or not chunks:
        chunks.append(cur)
    fixed: list[str] = []
    carry = False
    for c in chunks:
        if carry:
            c = "```" + c
        carry = c.count("```") % 2 == 1
        fixed.append(c + "```" if carry else c)
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": c or " "}} for c in fixed]
    if block_id:
        blocks[0]["block_id"] = block_id
    return blocks


def button(text: str, action_id: str, value: str, style: str | None = None) -> dict:
    b = {"type": "button", "action_id": action_id, "value": value, "text": {"type": "plain_text", "text": text[:75]}}
    if style:
        b["style"] = style
    return b


_SLACK_LINK = re.compile(r"<((?:https?|mailto|file):[^>|]+)(?:\|([^>]*))?>")


def slack_to_plain(text: str) -> str:
    """Undo Slack's message encoding so prompts/paths reach Claude verbatim."""
    # Prefer the label: it is what the user actually typed (Slack auto-links
    # things like "example.com" into <http://example.com|example.com>).
    text = _SLACK_LINK.sub(lambda m: m.group(2) or m.group(1).removeprefix("mailto:"), text)
    return html.unescape(text)


def inline_code(s: str) -> str:
    return "`" + escape(s.replace("`", "ʼ")) + "`"
