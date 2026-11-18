"""Persistent thread->session mapping (JSON, atomic writes) and audit log.

JSON + os.replace is used instead of SQLite because $HOME is on NFS, where
SQLite locking is unreliable. The bot is the only writer.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    channel: str
    thread_ts: str
    session_id: str
    cwd: str
    owner: str
    created: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    has_history: bool = False  # True once the CLI has written a transcript
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    fork_from: str | None = None  # first connect forks this session (fork_session=True)
    mode: str = "ask"  # "ask" = Slack approvals, "auto" = CLI auto-mode classifier
    model: str | None = None   # per-thread override of CLAUDE_MODEL
    effort: str | None = None  # per-thread override of DEFAULT_EFFORT (low|medium|high|xhigh|max)

    @property
    def key(self) -> str:
        return thread_key(self.channel, self.thread_ts)


def thread_key(channel: str, thread_ts: str) -> str:
    return f"{channel}:{thread_ts}"


def load_json_with_backup(path: Path) -> Any:
    """Parse ``path``; if it is corrupt, fall back to ``<path>.bak`` (written by
    ``_atomic_write(..., backup=True)``). A corrupt file is moved aside, never
    deleted. Returns None when there is nothing usable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text() or "{}")
    except (ValueError, OSError) as e:
        aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(path, aside)
        except OSError:
            aside = path
        bak = path.with_name(path.name + ".bak")
        try:
            data = json.loads(bak.read_text() or "{}")
            log.error("%s was unreadable (%s); moved to %s and restored the previous version from %s",
                      path.name, e, aside.name, bak.name)
            return data
        except (ValueError, OSError):
            log.error("%s was unreadable (%s) and has no usable backup; moved to %s, starting empty",
                      path.name, e, aside.name)
            return None


def _atomic_write(path: Path, data: str, backup: bool = False) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        if backup and path.exists():
            # Keep the previous good version (see load_json_with_backup).
            try:
                bak = path.with_name(path.name + ".bak")
                shutil.copyfile(path, bak)
                os.chmod(bak, 0o600)
            except OSError:
                log.warning("could not refresh %s.bak", path.name)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, SessionRecord] = {}
        raw = load_json_with_backup(path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    self.records[k] = SessionRecord(**v)
                except TypeError:
                    log.warning("skipping malformed session record %s", k)

    def save(self) -> None:
        data = {k: asdict(v) for k, v in self.records.items()}
        _atomic_write(self.path, json.dumps(data, indent=1), backup=True)

    def get(self, key: str) -> SessionRecord | None:
        return self.records.get(key)

    def put(self, rec: SessionRecord) -> None:
        self.records[rec.key] = rec
        self.save()

    def remove(self, key: str) -> None:
        if self.records.pop(key, None) is not None:
            self.save()

    def find_by_session(self, session_id: str) -> SessionRecord | None:
        for r in self.records.values():
            if r.session_id == session_id:
                return r
        return None


class AuditLog:
    """Append-only JSONL audit log of every tool call decision."""

    def __init__(self, path: Path, max_bytes: int = 20_000_000, keep: int = 5) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.keep = keep
        if not path.exists():
            path.touch(mode=0o600)
        os.chmod(path, 0o600)

    def _rotate(self) -> None:
        """audit.jsonl -> audit.jsonl.1 -> ... -> audit.jsonl.<keep> (oldest dropped)."""
        for i in range(self.keep - 1, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{i}")
            if src.exists():
                os.replace(src, self.path.with_name(f"{self.path.name}.{i + 1}"))
        os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        self.path.touch(mode=0o600)
        os.chmod(self.path, 0o600)

    def write(self, **fields: Any) -> None:
        entry = {"ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **fields}
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                self._rotate()
            with self.path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            log.exception("failed to write audit log")
