"""Reload-on-outside-change and activity-based idle timer (offline)."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path
from typing import Any

import slack_agent_controller.session as session_mod
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


class FakeClient:
    def __init__(self) -> None:
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


def _session(tmp: Path, **over: Any):
    hub = Hub(make_config(tmp, roots=[tmp], **over), FakeWeb())  # type: ignore[arg-type]
    sess = hub.create_session("D1", "1.0", tmp, "UOWNER")
    sess.rec.has_history = True
    sess.client = FakeClient()  # type: ignore[assignment]
    return sess


def _fake_size(monkeypatch, value: int | None):
    class Info:
        file_size = value

    monkeypatch.setattr(session_mod, "get_session_info", lambda sid: Info() if value is not None else None)


def test_no_change_keeps_process(tmp_path: Path, monkeypatch):
    async def run():
        s = _session(tmp_path)
        s._baseline = 1000
        _fake_size(monkeypatch, 1000)
        assert await s._reload_if_changed() is None
        assert s.client is not None
    asyncio.run(run())


def test_outside_write_reloads(tmp_path: Path, monkeypatch):
    async def run():
        s = _session(tmp_path)
        fake = s.client
        s._baseline = 1000
        s.bg_tasks["t1"] = "watch job"
        _fake_size(monkeypatch, 1500)
        note = await s._reload_if_changed()
        assert note and "outside Slack" in note and "background task" in note
        assert s.client is None and fake.disconnected and s._baseline == 1500
        assert '"session-reload"' in (s.cfg.state_dir / "audit.jsonl").read_text()
    asyncio.run(run())


def test_skips_when_nothing_to_compare(tmp_path: Path, monkeypatch):
    async def run():
        s = _session(tmp_path)
        _fake_size(monkeypatch, 1500)
        s._baseline = None                       # no turn has finished yet
        assert await s._reload_if_changed() is None
        s._baseline = 1000
        s.client = None                          # nothing in memory to go stale
        assert await s._reload_if_changed() is None
        s.client = FakeClient()                  # type: ignore[assignment]
        _fake_size(monkeypatch, None)            # lookup failed: do not reload
        assert await s._reload_if_changed() is None
        assert s.client is not None
    asyncio.run(run())


def test_idle_timer_counts_from_last_activity(tmp_path: Path):
    """The reaper measures idle time from the last activity, not from when the
    worker started waiting; background tasks keep a session alive."""
    IDLE = 2.0

    async def start(tag: str, age: float, bg: bool = False):
        hub = Hub(dataclasses.replace(make_config(tmp_path, roots=[tmp_path]), idle_disconnect_s=IDLE),
                  FakeWeb())  # type: ignore[arg-type]
        sess = hub.create_session("D1", tag, tmp_path, "UOWNER")
        sess.client = FakeClient()  # type: ignore[assignment]
        sess._last_activity -= age
        if bg:
            sess.bg_tasks["t"] = "watch job"
        sess.worker = asyncio.create_task(sess._worker())
        return sess

    async def stop(sess):
        sess.queue.put_nowait(None)
        await asyncio.wait_for(sess.worker, 5)

    async def run():
        a = await start("a", age=1.0)                # window ends 1.0 s from now
        b = await start("b", age=0.0)                # window ends 2.0 s from now
        c = await start("c", age=50.0, bg=True)      # long idle, but a background task is running
        await asyncio.sleep(0.5)
        assert not a.worker.done()
        await asyncio.sleep(1.0)                     # t = 1.5 s
        assert a.worker.done(), "a should be reaped 1.0 s in (counted from its last activity)"
        assert not b.worker.done()
        b._last_activity = time.monotonic()          # new activity at 1.5 s pushes b's deadline to 3.5 s
        await asyncio.sleep(1.0)                     # t = 2.5 s: b would be gone without the activity
        assert not b.worker.done() and not c.worker.done()
        await stop(b)
        await stop(c)

    asyncio.run(run())


def test_restored_cost_baseline(tmp_path: Path, monkeypatch):
    """A resumed CLI continues the session's cumulative cost from the
    transcript's last cost-state entry; that is the baseline for deltas."""
    import json as _json
    import uuid as _uuid

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    s = _session(tmp_path)
    s.rec.session_id = str(_uuid.uuid4())
    d = tmp_path / ".claude" / "projects" / "slug"
    d.mkdir(parents=True)
    lines = [{"type": "user"}, {"type": "cost-state", "totalCostUSD": 10.5}, {"type": "assistant"},
             {"type": "cost-state", "totalCostUSD": 56.245}]
    (d / f"{s.rec.session_id}.jsonl").write_text("\n".join(_json.dumps(x) for x in lines) + "\n")
    assert s._restored_cost() == 56.245
    s.rec.has_history = False
    assert s._restored_cost() == 0.0  # brand-new session
