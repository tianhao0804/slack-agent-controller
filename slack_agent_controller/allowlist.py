"""Bash commands you approved in Slack, remembered for next time.

A command runs without a card only when its exact text *and* the directory it
runs in match one you approved earlier, or it is one of a few fixed read-only
"seed" commands. Nothing is prefix- or pattern-matched, and the hard denials in
``policy.decide`` are evaluated first, so protected paths and credentials stay
blocked even for a remembered command.

Never remembered (these always ask):
* commands flagged with a warning (environment dumps, network, protected paths),
* commands that delete, move, kill or change permissions,
* commands whose effect depends on shell expansion (variables, wildcards,
  backticks), because the same text can do different things on a later run.

A remembered entry pins the command *text*, not the contents of the scripts it
runs: ``python run.py`` runs whatever ``run.py`` is at that moment. The file
lives in the protected state directory, so Claude cannot edit it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import policy
from .config import Config
from .store import _atomic_write

log = logging.getLogger(__name__)

MAX_COMMAND_CHARS = 4000
VERSION = 1

_DESTRUCTIVE = re.compile(
    r"(?<![\w-])(?:rm|rmdir|mv|kill|pkill|killall|scancel|scontrol|dd|shred|truncate|sudo|su|"
    r"chmod|chown|chgrp|reboot|shutdown|mkfs[\w.]*)(?![\w-])"
    r"|\bgit\s+(?:push|reset|clean|rebase|checkout|restore|stash|branch\s+-[dD])\b"
    r"|-delete\b|\brmtree\b|\bos\.remove\b|\bunlink\b"
)
_EXPANSION = re.compile(r"[*?`$]")


# Looking at jobs never needs a tap; submitting and cancelling do (they are not matched here).
JOB_READONLY = re.compile(r"(?:\S*/)?sac-job (?:list(?: --all)?|status [0-9a-f]{6}|log [0-9a-f]{6}(?: -n \d{1,4})?)")


def seed_commands() -> frozenset[str]:
    """Fixed read-only commands that never need a tap (exact text, any directory)."""
    cmds = {
        "nvidia-smi", "nvidia-smi -L",
        "git status", "git status -s", "git diff --stat", "git log --oneline -n 20", "git branch",
    }
    user = os.environ.get("USER", "")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        cmds.add(f"squeue -u {user}")
    return frozenset(cmds)


@dataclass
class Entry:
    key: str        # sha256 over the directory and the command text
    command: str    # as approved
    cwd: str        # directory the command ran in (resolved)
    added: float
    by: str         # Slack user who approved it
    session: str

    @property
    def id(self) -> str:
        return self.key[:8]


def make_key(command: str, cwd: Path) -> str:
    h = hashlib.sha256()
    h.update(os.path.realpath(cwd).encode())
    h.update(b"\0")
    h.update(command.strip().encode())
    return h.hexdigest()


class Allowlist:
    def __init__(self, path: Path, cfg: Config) -> None:
        self.path = path
        self.cfg = cfg
        self.seeds = seed_commands()
        self.entries: dict[str, Entry] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text() or "{}")
            items = raw.get("entries", []) if isinstance(raw, dict) else []
        except (ValueError, OSError) as e:
            aside = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(self.path, aside)
            except OSError:
                aside = self.path
            log.error("allowlist unreadable (%s); moved to %s and starting empty (everything asks again)", e, aside)
            return
        for item in items:
            try:
                ent = Entry(**item)
            except TypeError:
                log.warning("skipping malformed allowlist entry")
                continue
            self.entries[ent.key] = ent

    def save(self) -> None:
        data = {"version": VERSION, "entries": [asdict(e) for e in self.entries.values()]}
        _atomic_write(self.path, json.dumps(data, indent=1, ensure_ascii=False))

    # -- rules -------------------------------------------------------------
    def blocked_reason(self, tin: dict[str, Any], cwd: Path) -> str | None:
        """Why this command can never be remembered or matched (None = fine)."""
        cmd = str(tin.get("command", "")).strip()
        if not cmd:
            return "it is empty"
        if len(cmd) > MAX_COMMAND_CHARS:
            return "it is too long"
        if tin.get("dangerouslyDisableSandbox"):
            return "it asks to run outside the sandbox"
        try:
            flagged = bool(policy.bash_warnings(self.cfg, cmd, cwd)
                           or policy.output_protected(self.cfg, "Bash", {"command": cmd}, cwd))
        except (ValueError, OSError):  # e.g. NUL bytes in a path-like word: cannot be checked, so never match
            return "it could not be checked"
        if flagged:
            return "it touches the environment, the network or a protected path"
        if _DESTRUCTIVE.search(cmd):
            return "it deletes, moves or kills things or changes permissions"
        if _EXPANSION.search(cmd):
            return "its effect depends on shell expansion (variables, wildcards, backticks)"
        return None

    def allowed(self, tin: dict[str, Any], cwd: Path) -> str | None:
        """A short reason if this exact command may run without asking."""
        if self.blocked_reason(tin, cwd) is not None:
            return None
        cmd = str(tin.get("command", "")).strip()
        if cmd in self.seeds:
            return "built-in read-only command"
        if JOB_READONLY.fullmatch(cmd):
            return "built-in read-only command (sac-job)"
        if self.cfg.learn_approved_commands:
            ent = self.entries.get(make_key(cmd, cwd))
            if ent is not None:
                return f"you approved this exact command earlier (#{ent.id})"
        return None

    def learn(self, tin: dict[str, Any], cwd: Path, by: str, session: str) -> tuple[Entry | None, str]:
        """Remember an approved command. Returns (entry, "added" | "exists") or (None, why not)."""
        if not self.cfg.learn_approved_commands:
            return None, "remembering approved commands is turned off"
        why = self.blocked_reason(tin, cwd)
        if why:
            return None, why
        cmd = str(tin.get("command", "")).strip()
        if cmd in self.seeds:
            return None, "it is already allowed (built-in)"
        key = make_key(cmd, cwd)
        if key in self.entries:
            return self.entries[key], "exists"
        ent = Entry(key=key, command=cmd, cwd=os.path.realpath(cwd), added=time.time(), by=by, session=session)
        self.entries[key] = ent
        self.save()
        return ent, "added"

    def can_remember(self, tin: dict[str, Any], cwd: Path) -> bool:
        cmd = str(tin.get("command", "")).strip()
        return self.cfg.learn_approved_commands and cmd not in self.seeds and self.blocked_reason(tin, cwd) is None

    def card_note(self, tin: dict[str, Any], cwd: Path) -> str:
        """One line for the approval card about what approving will do."""
        if not self.cfg.learn_approved_commands:
            return ""
        cmd = str(tin.get("command", "")).strip()
        if cmd in self.seeds:
            return ""
        why = self.blocked_reason(tin, cwd)
        if why:
            return f":information_source: Not remembered, this one always asks: {why}."
        return ":bookmark: Run & remember here saves this exact command for this directory (`!allowlist` to review)."

    # -- management --------------------------------------------------------
    def remove(self, prefix: str) -> tuple[Entry | None, str]:
        prefix = prefix.strip().lower().lstrip("#")
        if len(prefix) < 4:
            return None, "Give at least 4 characters of the id shown by `!allowlist`."
        hits = [e for e in self.entries.values() if e.key.startswith(prefix)]
        if not hits:
            return None, f"No remembered command with id `{prefix}`."
        if len(hits) > 1:
            return None, f"`{prefix}` matches {len(hits)} entries; give more characters."
        del self.entries[hits[0].key]
        self.save()
        return hits[0], "removed"

    def clear(self) -> int:
        n = len(self.entries)
        self.entries.clear()
        self.save()
        return n
