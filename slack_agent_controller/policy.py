"""Tool-permission policy and human-readable tool summaries.

Decision model (enforced by a PreToolUse hook on *every* tool call, so it
cannot be shadowed by settings files or the CLI's built-in auto-allow rules):

* ``deny``  - target path is sensitive, or a write targets outside CWD_ROOTS
* ``allow`` - read-only tool whose target is inside CWD_ROOTS and whose
              search scope does not contain a sensitive path
* ``ask``   - everything else -> routed to ``can_use_tool`` -> Slack buttons
"""

from __future__ import annotations

import difflib
import fnmatch
import glob
import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import Config

Decision = Literal["allow", "deny", "ask"]

# Tools with no side effects. Their sub-calls (for Agent/Task) are still
# individually gated by the hook.
READ_ONLY_TOOLS = frozenset(
    {
        "Read",
        "Grep",
        "Glob",
        "LS",
        "NotebookRead",
        "TodoWrite",
        "TodoRead",
        "Agent",
        "Task",
        "TaskOutput",
        "TaskList",
        "TaskGet",
        "ToolSearch",
    }
)
# Tools whose primary target path must be checked.
WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
_PATH_KEYS = ("file_path", "notebook_path", "path")


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str
    # In auto mode an "ask" normally goes to the CLI classifier; this sends it to Slack instead.
    force_slack: bool = False


def _resolve(raw: str, cwd: Path) -> Path:
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = cwd / p
    return Path(os.path.realpath(p))


def is_sensitive(cfg: Config, p: Path) -> bool:
    for s in cfg.sensitive_paths:
        if p == s or p.is_relative_to(s):
            return True
    name = p.name
    return any(fnmatch.fnmatch(name, g) for g in cfg.sensitive_globs)


def contains_sensitive(cfg: Config, p: Path) -> bool:
    """True if directory ``p`` is an ancestor of some sensitive path."""
    return any(s.is_relative_to(p) for s in cfg.sensitive_paths)


def target_paths(tool: str, inp: dict[str, Any], cwd: Path) -> list[Path]:
    paths: list[Path] = []
    for k in _PATH_KEYS:
        v = inp.get(k)
        if isinstance(v, str) and v:
            paths.append(_resolve(v, cwd))
    if tool == "Glob":
        pat = inp.get("pattern")
        if isinstance(pat, str) and pat.startswith(("/", "~")):
            # Use the non-wildcard prefix of an absolute glob as its scope.
            prefix = re.split(r"[*?\[{]", pat, maxsplit=1)[0]
            paths.append(_resolve(prefix or "/", cwd))
    if tool in ("Grep", "Glob") and not inp.get("path"):
        paths.append(cwd)  # default search root
    return paths


# Bash subprocesses inherit the provider credential (verified for Bedrock: the CLI does not
# strip AWS_BEARER_TOKEN_BEDROCK, and its env-scrub mode needs a sandbox this
# node cannot provide). Commands naming it outright are denied; commands that
# could dump the environment get a warning on the approval card.
_BASH_DENY = re.compile(r"AWS_BEARER_TOKEN_BEDROCK|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|ANTHROPIC_API_KEY"
                        r"|ANTHROPIC_AUTH_TOKEN|/proc/[^\s/]+/environ")
_BASH_WARN = re.compile(
    r"(^|[;&|`(]\s*|\$\(\s*)(env|printenv|set|export\s+-p|declare\s+-[a-z]*x|compgen\s+-e)\s*($|[;&|)>])"
    r"|\bos\.environ\b|process\.env\b|\b(aws|curl|wget|nc|ncat|scp|rsync)\b"
)


# ---------------------------------------------------------------------------
# Shell words: braces, ~, $HOME and wildcards are expanded before checking, so
# `cat .e*` and `cat .{e,}nv` are seen as the `.env` they would read.
# ---------------------------------------------------------------------------
_MAX_ALTERNATIVES = 256
_MAX_GLOB_MATCHES = 500
_PATH_BUDGET = 2000
_WILDCARD = re.compile(r"[*?\[]")
_SEQ_NUM = re.compile(r"^(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?$")
_SEQ_CHR = re.compile(r"^([A-Za-z])\.\.([A-Za-z])(?:\.\.(-?\d+))?$")
_TOO_MANY = "uses an expansion with too many alternatives to check for protected paths"
_UNCHECKABLE = "has a word that could not be checked for protected paths"


class TooManyAlternatives(Exception):
    """A brace expression expands to more words than we are willing to check."""


def _sequence(body: str) -> list[str] | None:
    """`1..5`, `a..e`, `10..1..3` -> the words they stand for; None if not a sequence."""
    m = _SEQ_NUM.match(body)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        step = abs(int(m.group(3) or 1)) or 1
        if abs(hi - lo) // step + 1 > _MAX_ALTERNATIVES:
            raise TooManyAlternatives
        digits = [m.group(1).lstrip("-"), m.group(2).lstrip("-")]
        pad = max(map(len, digits)) if any(len(d) > 1 and d.startswith("0") for d in digits) else 0
        nums = range(lo, hi + 1, step) if lo <= hi else range(lo, hi - 1, -step)
        return [str(n).zfill(pad) if pad else str(n) for n in nums]
    m = _SEQ_CHR.match(body)
    if m:
        lo, hi = ord(m.group(1)), ord(m.group(2))
        step = abs(int(m.group(3) or 1)) or 1
        chars = range(lo, hi + 1, step) if lo <= hi else range(lo, hi - 1, -step)
        return [chr(c) for c in chars]
    return None


def _split_top_level(body: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for c in body:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return parts


def _match_brace(word: str, i: int) -> int:
    depth = 0
    for j in range(i, len(word)):
        if word[j] == "{":
            depth += 1
        elif word[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def brace_expand(word: str, limit: int = _MAX_ALTERNATIVES) -> list[str]:
    """Bash-style brace expansion (`{a,b}`, `{1..3}`, nested, empty alternatives).
    Raises TooManyAlternatives past ``limit`` results."""
    i = word.find("{")
    while i != -1:
        j = _match_brace(word, i)
        if j != -1:
            body = word[i + 1 : j]
            parts = _split_top_level(body)
            alts = parts if len(parts) > 1 else _sequence(body)
            if alts is not None:
                prefix, suffix = word[:i], word[j + 1 :]
                out: list[str] = []
                for alt in alts:
                    out.extend(brace_expand(prefix + alt + suffix, limit))
                    if len(out) > limit:
                        raise TooManyAlternatives
                return out
        i = word.find("{", i + 1)
    return [word]


def _bash_words(cmd: str) -> list[str]:
    """Words of a shell command that could name files. Options are dropped, but
    ``--opt=path`` / ``a=b`` values and a path glued to a short option (``-f.env``) are kept."""
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        words = list(lex)
    except ValueError:
        words = cmd.split()
    out = []
    for w in words:
        if "=" in w:
            w = w.split("=", 1)[1]
        w = w.strip("'\"")
        if not w or set(w) <= set("|&;<>()"):
            continue
        if w.startswith("-"):
            if not w.startswith("--") and len(w) > 2:
                out.append(w[2:])
            continue
        out.append(w)
    return out


def _expand_paths(word: str, cwd: Path) -> list[Path]:
    """Paths one brace-free shell word may name after ~, $HOME and wildcard expansion."""
    home = str(Path.home())
    w = os.path.expanduser(word.replace("${HOME}", home).replace("$HOME", home))
    if not _WILDCARD.search(w):
        return [Path(w)]
    pattern = w if os.path.isabs(w) else os.path.join(str(cwd), w)
    out: list[Path] = []
    try:
        for n, m in enumerate(glob.iglob(pattern)):
            if n >= _MAX_GLOB_MATCHES:
                break
            out.append(Path(m))
    except (OSError, ValueError, re.error):
        pass
    return out


def word_warnings(cfg: Config, command: str, cwd: Path) -> list[str]:
    """Protected files/dirs the command's words name once expanded (names such as
    ``.env`` and ``*.pem`` count, not only the configured paths)."""
    home = str(Path.home())
    warns: list[str] = []
    budget = _PATH_BUDGET
    for w in _bash_words(command):
        try:
            alts = brace_expand(w)
        except TooManyAlternatives:
            if _TOO_MANY not in warns:
                warns.append(_TOO_MANY)
            continue
        for alt in alts:
            for path in _expand_paths(alt, cwd):
                budget -= 1
                if budget < 0:
                    if _TOO_MANY not in warns:
                        warns.append(_TOO_MANY)
                    return warns
                try:
                    rp = _resolve(str(path), cwd)
                except (OSError, ValueError):  # cannot be checked: treat as protected (fail closed)
                    if _UNCHECKABLE not in warns:
                        warns.append(_UNCHECKABLE)
                    continue
                if is_sensitive(cfg, rp):
                    msg = f"names protected path {str(rp).replace(home, '~', 1)}"
                    if msg not in warns:
                        warns.append(msg)
    return warns


def bash_warnings(cfg: Config, command: str, cwd: Path | None = None) -> list[str]:
    """Reasons a Bash command must be confirmed in Slack even in auto mode.
    With ``cwd`` the command's words are also expanded and checked against protected files."""
    warns: list[str] = []
    if _BASH_WARN.search(command):
        warns.append("may expose environment variables (incl. the model provider's API key) or send data off-node")
    home = str(Path.home())
    for s in cfg.sensitive_paths:
        sp = str(s)
        if sp in command or sp.replace(home, "~", 1) in command or sp.replace(home, "$HOME", 1) in command:
            warns.append(f"references protected path {sp.replace(home, '~', 1)}")
    if cwd is not None:
        warns += [w for w in word_warnings(cfg, command, cwd) if w not in warns]
    return warns


def _protected_name_samples(cfg: Config) -> list[str]:
    """Names a protected file can have: basenames of protected paths, literal
    globs (``.env``) and one sample per wildcard glob (``*.pem`` -> ``x.pem``)."""
    names = [s.name for s in cfg.sensitive_paths if s.name]
    names += [re.sub(r"[*?]", "x", g) for g in cfg.sensitive_globs]
    return names


def glob_targets_protected(cfg: Config, pattern: str) -> bool:
    """True if a search glob filter (Grep's ``glob``) can select protected files."""
    samples = _protected_name_samples(cfg)
    for tok in pattern.split():
        if tok.startswith("!"):
            continue  # an exclusion
        try:
            alts = brace_expand(tok)
        except TooManyAlternatives:
            return True
        for alt in alts:
            base = alt.rsplit("/", 1)[-1]
            if base and (any(fnmatch.fnmatch(base, g) for g in cfg.sensitive_globs)
                         or any(fnmatch.fnmatch(n, base) for n in samples)):
                return True
    return False


MEMORY_TOOLS = frozenset({"Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "LS"})


def project_memory_dir(cwd: str | Path) -> Path:
    """Claude Code's auto-memory directory for a session started in ``cwd`` (as given):
    ~/.claude/projects/<cwd with every non-alphanumeric character as '-'>/memory."""
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd)) / "memory"


def in_memory_dir(p: Path, memory_dir: Path | None) -> bool:
    """``p`` (resolved) is inside this session's own memory directory, which itself must
    resolve inside ~/.claude/projects (a symlinked memory dir pointing elsewhere does not count)."""
    if memory_dir is None:
        return False
    projects = Path(os.path.realpath(Path.home() / ".claude" / "projects"))
    real_mem = Path(os.path.realpath(memory_dir))
    if real_mem.parent.parent != projects or real_mem.name != "memory":
        return False
    return p == real_mem or p.is_relative_to(real_mem)


def decide(cfg: Config, tool: str, inp: dict[str, Any], cwd: Path, memory_dir: Path | None = None) -> PolicyResult:
    if tool == "Bash" and _BASH_DENY.search(str(inp.get("command", ""))):
        return PolicyResult("deny", "Commands that reference cloud credentials or process environments are blocked by the Slack bot policy.")
    paths = target_paths(tool, inp, cwd)
    if tool in MEMORY_TOOLS and paths and memory_dir is not None and all(in_memory_dir(p, memory_dir) for p in paths):
        # The user opened this project's auto-memory notes (and only those) to the session.
        return PolicyResult("allow", "this project's auto-memory directory")
    for p in paths:
        if is_sensitive(cfg, p):
            return PolicyResult("deny", f"{p} is a protected path (secrets/config); access blocked by the Slack bot policy.")

    if tool in WRITE_TOOLS:
        for p in paths:
            if not cfg.under_roots(p):
                return PolicyResult("deny", f"{p} is outside the allowed roots; writes are blocked.")
        return PolicyResult("ask", "file modification requires approval")

    if tool in READ_ONLY_TOOLS or tool in cfg.auto_allow_tools:
        for p in paths:
            if not cfg.readable(p):
                return PolicyResult("ask", f"{p} is outside the allowed roots")
            if tool in ("Grep", "Glob", "LS") and contains_sensitive(cfg, p):
                return PolicyResult("ask", f"search scope {p} contains protected paths", force_slack=True)
        if tool == "Grep" and glob_targets_protected(cfg, str(inp.get("glob") or "")):
            return PolicyResult("ask", "the glob filter can select protected files", force_slack=True)
        return PolicyResult("allow", "read-only tool within allowed roots")

    return PolicyResult("ask", "tool has side effects")


# ---------------------------------------------------------------------------
# Summaries for Slack and the audit log
# ---------------------------------------------------------------------------


def _short(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _rel(raw: str, cwd: Path) -> str:
    try:
        p = _resolve(raw, cwd)
        return str(p.relative_to(cwd)) if p.is_relative_to(cwd) else str(p)
    except Exception:
        return raw


def tool_line(tool: str, inp: dict[str, Any], cwd: Path) -> str:
    """One-line description, e.g. ``Bash: nvidia-smi``."""
    if tool == "Bash":
        arg = _short(str(inp.get("command", "")), 110)
    elif tool in ("Read", "Edit", "Write", "MultiEdit"):
        arg = _rel(str(inp.get("file_path", "")), cwd)
    elif tool == "NotebookEdit":
        arg = _rel(str(inp.get("notebook_path", "")), cwd)
    elif tool == "Grep":
        arg = _short(f"{inp.get('pattern', '')}  in {inp.get('path') or '.'}", 110)
    elif tool == "Glob":
        arg = _short(str(inp.get("pattern", "")), 110)
    elif tool in ("WebFetch",):
        arg = _short(str(inp.get("url", "")), 110)
    elif tool == "WebSearch":
        arg = _short(str(inp.get("query", "")), 110)
    elif tool in ("Agent", "Task"):
        arg = _short(str(inp.get("description", "")), 110)
    elif tool == "TodoWrite":
        arg = f"{len(inp.get('todos', []) or [])} items"
    else:
        arg = _short(json.dumps(inp, ensure_ascii=False, default=str), 90)
    return f"{tool}: {arg}" if arg else tool


def _diff(old: str, new: str, path: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=path, tofile=path, lineterm="", n=2
    )
    return "\n".join(lines)


def approval_full(tool: str, inp: dict[str, Any], cwd: Path) -> str:
    """The complete request, attached to the approval card when the preview is cut."""
    if tool == "Write":
        content = str(inp.get("content", ""))
        return _write_header(inp, cwd) + content
    return approval_detail(tool, inp, cwd)


WRITE_PREVIEW_LINES = 60


def _write_header(inp: dict[str, Any], cwd: Path) -> str:
    raw = str(inp.get("file_path", ""))
    content = str(inp.get("content", ""))
    state = "overwrite" if _resolve(raw, cwd).exists() else "new file"
    return f"{state}: {_rel(raw, cwd)} ({len(content.splitlines())} lines, {len(content)} chars)\n---\n"


def approval_detail(tool: str, inp: dict[str, Any], cwd: Path) -> str:
    """Full detail shown in the approval card (command text or diff)."""
    if tool == "Bash":
        desc = inp.get("description")
        cmd = str(inp.get("command", ""))
        extra = " (background)" if inp.get("run_in_background") else ""
        return (f"# {desc}{extra}\n" if desc else "") + cmd
    if tool == "Edit":
        path = _rel(str(inp.get("file_path", "")), cwd)
        d = _diff(str(inp.get("old_string", "")), str(inp.get("new_string", "")), path)
        return ("[replace_all]\n" if inp.get("replace_all") else "") + d
    if tool == "MultiEdit":
        path = _rel(str(inp.get("file_path", "")), cwd)
        parts = [
            _diff(str(e.get("old_string", "")), str(e.get("new_string", "")), path)
            for e in inp.get("edits", []) or []
        ]
        return "\n".join(parts)
    if tool == "Write":
        content = str(inp.get("content", ""))
        lines = content.splitlines(keepends=True)
        if len(lines) <= WRITE_PREVIEW_LINES:
            return _write_header(inp, cwd) + content
        rest = len(lines) - WRITE_PREVIEW_LINES
        return (_write_header(inp, cwd) + "".join(lines[:WRITE_PREVIEW_LINES])
                + f"\n…({rest} more line(s) in the attached file)…")
    if tool == "NotebookEdit":
        return (
            f"{_rel(str(inp.get('notebook_path', '')), cwd)} cell={inp.get('cell_id')} "
            f"mode={inp.get('edit_mode', 'replace')}\n---\n{inp.get('new_source', '')}"
        )
    if tool == "WebFetch":
        return f"URL: {inp.get('url')}\nPrompt: {inp.get('prompt', '')}"
    return json.dumps(inp, indent=1, ensure_ascii=False, default=str)


_SECRETISH = re.compile(
    r"(?i)((?:token|secret|password|passwd|api[_-]?key|bearer)\S*?\s*[=:]\s*)(\S+)"
)
_LONG_B64 = re.compile(r"[A-Za-z0-9+/=_-]{40,}")
# Well-known credential formats (redacted regardless of surrounding text).
_KNOWN_TOKENS = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"
    r"|\bABSK[A-Za-z0-9+/=]{16,}"                # Bedrock long-term API key
    r"|\bbedrock-api-key-[A-Za-z0-9+/=_-]{16,}"  # Bedrock short-term API key
    r"|\bxox[abposr]-[A-Za-z0-9-]{10,}"          # Slack tokens
    r"|\bxapp-[A-Za-z0-9-]{10,}"
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"            # AWS access key ids
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"             # GitHub tokens
    r"|\bsk-ant-[A-Za-z0-9_-]{20,}"               # Anthropic API keys
    r"|\bsk-[A-Za-z0-9_-]{20,}",                 # generic API keys
    re.S,
)


def redact(s: str) -> str:
    s = _KNOWN_TOKENS.sub("<redacted>", s)
    s = _SECRETISH.sub(lambda m: m.group(1) + "<redacted>", s)
    return _LONG_B64.sub("<redacted-blob>", s)


# Random-looking blobs: long, mixed case + digits, and not path-like. Unlike
# _LONG_B64 this keeps file paths, git hashes and identifiers readable, which
# matters for full terminal logs.
_BLOB = re.compile(r"[A-Za-z0-9+/=_-]{40,}")


def _entropy(t: str) -> float:
    n = len(t)
    return -sum(c / n * math.log2(c / n) for c in (t.count(ch) for ch in set(t)))


def _blob_sub(m: re.Match[str]) -> str:
    t = m.group(0)
    if t.count("/") >= 3:
        return t  # path
    for seg in re.split(r"[_.-]", t):
        if (len(seg) >= 20 and re.search(r"[a-z]", seg) and re.search(r"[A-Z]", seg)
                and re.search(r"\d", seg) and _entropy(seg) >= 4.0):
            return "<redacted-blob>"
    return t  # structured identifier (file names, AFLOW labels, hashes, ...)


def redact_log(s: str) -> str:
    """Redaction for full logs (terminal mirror): known credential formats,
    ``key=value`` secrets and random-looking blobs; paths stay intact."""
    s = _KNOWN_TOKENS.sub("<redacted>", s)
    s = _SECRETISH.sub(lambda m: m.group(1) + "<redacted>", s)
    return _BLOB.sub(_blob_sub, s)


def output_protected(cfg: Config, tool: str, inp: dict[str, Any], cwd: Path) -> bool:
    """True if a tool call's output (or written content) may expose protected
    data: it targets a protected path, or is a Bash command that references
    credentials or protected files (after brace/wildcard expansion)."""
    if tool == "Bash":
        cmd = str(inp.get("command", ""))
        return bool(_BASH_DENY.search(cmd)) or any("protected path" in w for w in bash_warnings(cfg, cmd, cwd))
    paths = target_paths(tool, inp, cwd)
    if any(is_sensitive(cfg, p) for p in paths):
        return True
    if tool == "Grep" and glob_targets_protected(cfg, str(inp.get("glob") or "")):
        return True
    return tool == "Grep" and any(contains_sensitive(cfg, p) for p in paths)


def audit_summary(tool: str, inp: dict[str, Any], limit: int = 1000) -> str:
    s = json.dumps(inp, ensure_ascii=False, default=str)
    return redact(s)[:limit]
