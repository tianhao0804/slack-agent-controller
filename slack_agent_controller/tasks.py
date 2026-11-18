"""Messages the bot has accepted but not finished, kept on disk (review R3).

A record is written when a Slack message is queued, marked ``running`` when it
is sent to Claude and removed when the reply has ended. If the bot stops in
between (restart, crash, node reboot), the next start tells each affected
thread what happened. Nothing is replayed automatically: an interrupted
reply may already have run commands, so only the user can decide to resend.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .store import _atomic_write, load_json_with_backup

log = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    id: str
    channel: str
    thread_ts: str
    session_id: str
    user: str
    text: str          # first characters of the message
    status: str        # "queued" | "running" | "interrupted"
    created: float
    started: float | None = None


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, TaskRecord] = {}
        raw = load_json_with_backup(path)
        for item in (raw.get("tasks", []) if isinstance(raw, dict) else []):
            try:
                rec = TaskRecord(**item)
            except TypeError:
                log.warning("skipping malformed task record")
                continue
            self.records[rec.id] = rec

    def save(self) -> None:
        data = {"version": 1, "tasks": [asdict(r) for r in self.records.values()]}
        try:
            _atomic_write(self.path, json.dumps(data, indent=1, ensure_ascii=False), backup=True)
        except OSError:
            log.exception("could not save %s", self.path.name)

    def add(self, channel: str, thread_ts: str, session_id: str, user: str, text: str) -> str:
        tid = uuid.uuid4().hex[:12]
        self.records[tid] = TaskRecord(id=tid, channel=channel, thread_ts=thread_ts, session_id=session_id,
                                       user=user, text=text[:500], status="queued", created=time.time())
        self.save()
        return tid

    def mark(self, tid: str | None, status: str) -> None:
        rec = self.records.get(tid or "")
        if rec is None:
            return
        rec.status = status
        if status == "running":
            rec.started = time.time()
        self.save()

    def done(self, *tids: str | None) -> None:
        removed = [t for t in tids if t and self.records.pop(t, None) is not None]
        if removed:
            self.save()

    def take_all(self) -> list[TaskRecord]:
        """Every record left from a previous run (oldest first); the store is emptied."""
        recs = sorted(self.records.values(), key=lambda r: r.created)
        if recs:
            self.records.clear()
            self.save()
        return recs
