"""Render Claude Code transcript entries (JSONL) as a Markdown log.

Used by the terminal mirror: everything new in a session's transcript since
the last turn is rendered into one ``turn-NNN.md`` file — prompts, Claude's
text and thinking, every tool call with its full input and output, and
sub-agent activity. Long blocks are clipped (head + tail), outputs of tool
calls that touch protected paths are omitted, and the whole document goes
through ``policy.redact_log``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import policy
from .config import Config


@dataclass(frozen=True)
class Limits:
    head: int = 200          # lines kept at the start of a long block
    tail: int = 200          # ... and at the end
    block_chars: int = 60_000


DEFAULT_LIMITS = Limits()
SMALL_LIMITS = Limits(head=50, tail=50, block_chars=15_000)
MAX_FILE_BYTES = 1_000_000


def read_new(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Complete JSONL lines after ``offset``. Returns (entries, new_offset);
    a partially written last line is left for next time."""
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    entries = []
    for line in data[: end + 1].splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            entries.append(e)
    return entries, offset + end + 1


def clip(s: str, lim: Limits) -> str:
    lines = s.split("\n")
    if len(lines) > lim.head + lim.tail + 10:
        cut = len(lines) - lim.head - lim.tail
        s = "\n".join(lines[: lim.head] + [f"… [{cut} lines omitted] …"] + lines[-lim.tail :])
    if len(s) > lim.block_chars:
        half = lim.block_chars // 2
        s = s[:half] + f"\n… [{len(s) - 2 * half} characters omitted] …\n" + s[-half:]
    return s


def fence(s: str, lang: str = "") -> str:
    runs = [len(m) for m in re.findall(r"`{3,}", s)]
    tick = "`" * max(3, max(runs, default=0) + 1)
    return f"{tick}{lang}\n{s.rstrip()}\n{tick}"


def _clock(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


def _parse_ts(ts: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
    except ValueError:
        return None


def _content_text(content: Any) -> str:
    """Text of a tool_result / message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            parts.append(str(b.get("text", "")))
        elif t == "image":
            parts.append("[image]")
        elif t == "tool_reference":
            parts.append(f"[tool reference: {b.get('tool_name', '')}]")
        else:
            parts.append(f"[{t}]")
    return "\n".join(parts)


@dataclass
class Rendered:
    markdown: str = ""
    prompts: list[str] = field(default_factory=list)  # your prompts in this chunk (redacted)
    final_text: str = ""                              # Claude's last text block (redacted)
    tool_calls: int = 0
    subagents: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    started: dt.datetime | None = None
    ended: dt.datetime | None = None
    models: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float | None:
        if self.started and self.ended:
            return max(0.0, (self.ended - self.started).total_seconds())
        return None

    @property
    def empty(self) -> bool:
        return not (self.prompts or self.final_text or self.tool_calls or self.subagents)


@dataclass
class SubagentChunk:
    agent_id: str
    description: str
    tool_use_id: str | None
    finished: bool
    entries: list[dict[str, Any]]


class _Writer:
    def __init__(self, cfg: Config, cwd: Path, lim: Limits, res: Rendered) -> None:
        self.cfg, self.cwd, self.lim, self.res = cfg, cwd, lim, res
        self.out: list[str] = []
        self.calls: dict[str, tuple[str, str, bool]] = {}  # tool_use_id -> (label, name, protected)
        self.seen_msg_ids: set[str] = set()

    def emit(self, s: str) -> None:
        self.out.append(s)

    def entries(self, entries: list[dict[str, Any]], h: int, prefix: str, main: bool) -> None:
        role = None
        n = 0
        for e in entries:
            t = e.get("type")
            ts = e.get("timestamp")
            if main and ts:
                when = _parse_ts(ts)
                if when:
                    self.res.started = self.res.started or when
                    self.res.ended = when
            if t == "system":
                st = e.get("subtype")
                if st == "compact_boundary":
                    self.emit(f"\n---\n_{_clock(ts)} context compacted_\n")
                elif st == "local_command" and e.get("content"):
                    self.emit(f"\n{'#' * h} ⌨️ Local command {_clock(ts)}\n\n" + fence(clip(str(e["content"]), self.lim)))
                continue
            if t not in ("user", "assistant") or (main and e.get("isSidechain")):
                continue
            msg = e.get("message") or {}
            content = msg.get("content")
            if t == "assistant":
                self._usage(msg)
                if role != "assistant":
                    self.emit(f"\n{'#' * h} 🤖 Claude {_clock(ts)}")
                    role = "assistant"
                for b in content if isinstance(content, list) else []:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text" and str(b.get("text", "")).strip():
                        text = str(b["text"])
                        self.emit("\n" + text.strip())
                        if main:
                            self.res.final_text = text
                    elif bt == "thinking" and str(b.get("thinking", "")).strip():
                        body = clip(str(b["thinking"]).strip(), self.lim)
                        self.emit("\n> 💭 *thinking*\n" + "\n".join("> " + ln for ln in body.splitlines()))
                    elif bt == "tool_use":
                        n += 1
                        self._tool_use(b, f"{prefix}{n}", h + 1, ts, main)
                continue
            # user entry: tool results, prompts, or text inserted by Claude Code
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        self._tool_result(b, h + 1)
                texts = [str(b.get("text", "")) for b in content
                         if isinstance(b, dict) and b.get("type") == "text" and str(b.get("text", "")).strip()]
                if any(isinstance(b, dict) and b.get("type") == "image" for b in content):
                    texts.append("[image]")
            else:
                texts = [content] if isinstance(content, str) and content.strip() else []
            if texts:
                role = "user"
                self._user_text("\n".join(texts), e, h, ts, main)

    def _usage(self, msg: dict[str, Any]) -> None:
        mid = msg.get("id")
        if mid and mid in self.seen_msg_ids:
            return  # every content block of one API message repeats the same usage
        if mid:
            self.seen_msg_ids.add(mid)
        u = msg.get("usage") or {}
        self.res.tokens_in += sum(int(u.get(k) or 0) for k in
                                  ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        self.res.tokens_out += int(u.get("output_tokens") or 0)
        model = msg.get("model")
        if model and model not in self.res.models and not str(model).startswith("<"):
            self.res.models.append(str(model))

    def _user_text(self, text: str, e: dict[str, Any], h: int, ts: str | None, main: bool) -> None:
        kind = (e.get("origin") or {}).get("kind") if isinstance(e.get("origin"), dict) else None
        stripped = text.lstrip()
        tag = re.match(r"<([\w-]+)", stripped)
        if kind == "task-notification":
            head = "⚙️ Background task notification"
        elif kind == "peer":
            head = "📨 Message from a sub-agent"
        elif e.get("isMeta") or tag or stripped.startswith("Caveat:"):
            head = f"📎 Inserted by Claude Code ({tag.group(1)})" if tag else "📎 Inserted by Claude Code"
        else:
            head = "👤 You" if main else "📝 Task given to the sub-agent"
            if main:
                self.res.prompts.append(text)
            self.emit(f"\n{'#' * h} {head} {_clock(ts)}\n\n" + clip(text.strip(), self.lim))
            return
        self.emit(f"\n{'#' * h} {head} {_clock(ts)}\n\n" + fence(clip(text.strip(), self.lim)))

    def _tool_use(self, b: dict[str, Any], label: str, h: int, ts: str | None, main: bool) -> None:
        name = str(b.get("name", "?"))
        raw = b.get("input")
        inp: dict[str, Any] = raw if isinstance(raw, dict) else {}
        protected = policy.output_protected(self.cfg, name, inp, self.cwd)
        self.calls[str(b.get("id", ""))] = (label, name, protected)
        if main:
            self.res.tool_calls += 1
        line = policy.tool_line(name, inp, self.cwd)
        self.emit(f"\n{'#' * h} 🔧 #{label} {line.splitlines()[0] if line else name} {_clock(ts)}")
        if protected and name in policy.WRITE_TOOLS:
            self.emit("\n_[content omitted: protected path]_")
            return
        if name == "Bash":
            desc = inp.get("description")
            if desc:
                self.emit(f"\n_{desc}_" + (" _(background)_" if inp.get("run_in_background") else ""))
            self.emit("\n" + fence(clip(str(inp.get("command", "")), self.lim), "bash"))
        elif name in ("Edit", "MultiEdit"):
            self.emit("\n" + fence(clip(policy.approval_detail(name, inp, self.cwd), self.lim), "diff"))
        elif name == "Write":
            self.emit(f"\n`{inp.get('file_path', '')}`\n\n" + fence(clip(str(inp.get("content", "")), self.lim)))
        elif name == "Read":
            extra = " ".join(f"{k}={inp[k]}" for k in ("offset", "limit", "pages") if inp.get(k))
            if extra:
                self.emit(f"\n_{extra}_")
        else:
            body = json.dumps(inp, indent=1, ensure_ascii=False, default=str)
            self.emit("\n" + fence(clip(body, self.lim), "json"))

    def _tool_result(self, b: dict[str, Any], h: int) -> None:
        label, _name, protected = self.calls.get(str(b.get("tool_use_id", "")), ("?", "tool", False))
        err = " (error)" if b.get("is_error") else ""
        if protected:
            self.emit(f"\n**Output of #{label}{err}:** _[omitted: touches a protected path]_")
            return
        text = _content_text(b.get("content")).strip()
        if not text:
            self.emit(f"\n**Output of #{label}{err}:** _(empty)_")
            return
        self.emit(f"\n**Output of #{label}{err}:**\n" + fence(clip(text, self.lim)))


def render(cfg: Config, cwd: Path, title: str, entries: list[dict[str, Any]],
           subagents: list[SubagentChunk] = (), lim: Limits = DEFAULT_LIMITS) -> Rendered:  # type: ignore[assignment]
    res = Rendered()
    w = _Writer(cfg, cwd, lim, res)
    w.entries(entries, 2, "", main=True)
    for i, sa in enumerate(subagents, 1):
        if not any(e.get("type") in ("user", "assistant") for e in sa.entries):
            continue
        res.subagents += 1
        launched = w.calls.get(sa.tool_use_id or "")
        origin = f" (launched by #{launched[0]})" if launched else ""
        state = "" if sa.finished else " — still running, continued in a later turn"
        w.emit(f"\n\n---\n\n## 🧩 Sub-agent: {sa.description or sa.agent_id}{origin}{state}")
        w.entries(sa.entries, 3, f"a{i}.", main=False)

    head = [f"# {title}", ""]
    meta = [f"- cwd: `{cwd}`"]
    if res.started:
        span = res.started.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        if res.ended:
            span += " → " + res.ended.astimezone().strftime("%H:%M:%S")
        meta.append(f"- time: {span}")
    if res.models:
        meta.append(f"- model: {', '.join(res.models)}")
    meta.append(f"- tool calls: {res.tool_calls} · sub-agents: {res.subagents} · "
                f"tokens in {res.tokens_in:,} / out {res.tokens_out:,}")
    meta.append("- outputs of tool calls touching protected paths are omitted; secrets redacted; long blocks clipped")
    res.markdown = policy.redact_log("\n".join(head + meta) + "\n" + "\n".join(w.out).strip() + "\n")
    res.prompts = [policy.redact_log(p) for p in res.prompts]
    res.final_text = policy.redact_log(res.final_text)
    return res


def render_capped(cfg: Config, cwd: Path, title: str, entries: list[dict[str, Any]],
                  subagents: list[SubagentChunk] = ()) -> Rendered:  # type: ignore[assignment]
    """``render`` with the 1 MB file cap: retry with tighter clipping, then cut."""
    res = render(cfg, cwd, title, entries, subagents)
    if len(res.markdown.encode()) > MAX_FILE_BYTES:
        res = render(cfg, cwd, title, entries, subagents, SMALL_LIMITS)
    raw = res.markdown.encode()
    if len(raw) > MAX_FILE_BYTES:
        keep = MAX_FILE_BYTES - 200
        res.markdown = (raw[: keep // 2].decode(errors="ignore")
                        + f"\n\n… [{len(raw) - keep} bytes omitted to stay under 1 MB] …\n\n"
                        + raw[-keep // 2:].decode(errors="ignore"))
    return res
