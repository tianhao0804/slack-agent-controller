"""Reliability fixes from the review's table: state-file recovery, audit rotation, single
instance, concurrency/queue limits, failure notifications."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.hub import Hub
from slack_agent_controller.mirror import Mirror
from slack_agent_controller.session import Prompt, TurnRenderer, _Turn
from slack_agent_controller.store import AuditLog, SessionStore
from tests.helpers import FakeWeb, make_config

ROOT = Path(__file__).resolve().parent.parent


def test_sessions_json_recovers_from_backup(tmp_path: Path):
    hub = Hub(make_config(tmp_path, [tmp_path]), FakeWeb())  # type: ignore[arg-type]
    hub.create_session("D1", "1.0", tmp_path, "UOWNER")
    hub.create_session("D1", "2.0", tmp_path, "UOWNER")       # second save -> .bak holds the first state
    path = hub.store.path
    assert path.with_name("sessions.json.bak").exists()
    path.write_text("{ corrupt")
    store = SessionStore(path)
    assert "D1:1.0" in store.records                           # restored from the backup
    assert list(path.parent.glob("sessions.json.corrupt-*")), "corrupt file kept for inspection"


def test_sessions_json_without_backup_starts_empty_but_keeps_the_file(tmp_path: Path):
    path = tmp_path / "sessions.json"
    path.write_text("{ corrupt")
    store = SessionStore(path)
    assert store.records == {}
    assert list(tmp_path.glob("sessions.json.corrupt-*"))


def test_mirror_json_recovers_from_backup(tmp_path: Path):
    hub = Hub(make_config(tmp_path, [tmp_path]), FakeWeb())  # type: ignore[arg-type]
    m = Mirror(hub)
    from slack_agent_controller.mirror import MirrorState
    m.states["a"] = MirrorState(session_id="a", cwd=str(tmp_path), transcript="t")
    m.save()
    m.save()
    m.path.write_text("not json")
    assert "a" in Mirror(hub).states


def test_audit_log_rotates(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl", max_bytes=200, keep=3)
    for i in range(40):
        log.write(event="x", i=i, pad="y" * 30)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["audit.jsonl", "audit.jsonl.1", "audit.jsonl.2", "audit.jsonl.3"], names
    for p in tmp_path.iterdir():
        assert oct(p.stat().st_mode & 0o777) == "0o600"
    last = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert last["i"] == 39


def test_second_instance_refuses_to_start(tmp_path: Path):
    code = ("import sys; sys.path.insert(0, %r); from bot import acquire_instance_lock; from pathlib import Path; "
            "import time; l = acquire_instance_lock(Path(%r), lock_dir=Path(%r)); print('got' if l else 'refused', flush=True); "
            "time.sleep(%s)")
    first = subprocess.Popen([sys.executable, "-c", code % (str(ROOT), str(tmp_path), str(tmp_path), 5)], stdout=subprocess.PIPE,
                             text=True)
    assert first.stdout and first.stdout.readline().strip() == "got"
    second = subprocess.run([sys.executable, "-c", code % (str(ROOT), str(tmp_path), str(tmp_path), 0)], capture_output=True,
                            text=True, timeout=20)
    first.kill()
    first.wait()
    assert second.stdout.strip() == "refused" and "already running" in second.stderr
    third = subprocess.run([sys.executable, "-c", code % (str(ROOT), str(tmp_path), str(tmp_path), 0)], capture_output=True,
                           text=True, timeout=20)
    assert third.stdout.strip() == "got", "lock is released when the holder exits"


def test_queue_limit_per_thread(tmp_path: Path):
    async def run():
        cfg = dataclasses.replace(make_config(tmp_path, [tmp_path]), max_queued_per_thread=2)
        f = SlackFrontend(cfg)
        f.hub.web = FakeWeb()  # type: ignore[assignment]
        s = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.worker = asyncio.create_task(asyncio.sleep(10))      # pretend the worker is busy
        for i in range(3):
            await f._prompt("D1", f"5.{i}", "1.0", "UOWNER", f"msg {i}")
        assert s.queue.qsize() == 2
        assert any("this one was not queued" in t for t in f.hub.web.texts())
        s.worker.cancel()
    asyncio.run(run())


def test_running_turns_are_limited(tmp_path: Path):
    async def run():
        cfg = dataclasses.replace(make_config(tmp_path, [tmp_path]), max_running_turns=1)
        hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
        a = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        b = hub.create_session("D1", "2.0", tmp_path, "UOWNER")
        gate = asyncio.Event()
        started: list[str] = []

        async def fake_turn(self, item, note):  # noqa: ANN001
            started.append(item.text)
            await gate.wait()

        import types
        for s in (a, b):
            s._run_turn = types.MethodType(fake_turn, s)  # type: ignore[method-assign]
            s._reload_if_changed = AsyncMock(return_value=None)  # type: ignore[method-assign]
        ta = asyncio.create_task(a._run_prompt(Prompt("first", "UOWNER")))
        await asyncio.sleep(0.05)
        tb = asyncio.create_task(b._run_prompt(Prompt("second", "UOWNER")))
        await asyncio.sleep(0.05)
        assert started == ["first"], "the second waits for a free slot"
        assert any("Waiting: " in t for t in hub.web.texts())
        gate.set()
        await asyncio.wait_for(asyncio.gather(ta, tb), 2)
        assert started == ["first", "second"]
    asyncio.run(run())


def _finish(tmp_path: Path, result, error=None, autonomous=False, elapsed=1.0) -> list[str]:  # noqa: ANN001
    async def run():
        hub = Hub(make_config(tmp_path, [tmp_path]), FakeWeb())  # type: ignore[arg-type]
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        r = TurnRenderer(hub, "D1", "1.0", tmp_path)
        r.ts = "9.0"
        t = _Turn(r, "UOWNER", time.monotonic(), autonomous)
        t.result, t.error = result, error
        await s._finish_turn(t, elapsed)
        return [kw["text"] for op, kw in hub.web.calls if op == "post"]
    return asyncio.run(run())


class _Result:
    def __init__(self, is_error=False, terminal_reason="completed", subtype="success"):  # noqa: ANN001
        self.is_error, self.terminal_reason, self.subtype = is_error, terminal_reason, subtype
        self.total_cost_usd, self.usage, self.session_id = 0.01, {}, None
        self.api_error_status, self.num_turns, self.errors = None, 1, ["error_max_budget_usd"]


def test_short_failures_always_notify_with_a_red_mark(tmp_path: Path):
    posts = _finish(tmp_path, _Result(is_error=True, subtype="error_max_budget_usd"))
    assert any("<@UOWNER> :x: failed" in p for p in posts), posts
    assert not any(":white_check_mark:" in p for p in posts)


def test_short_success_does_not_ping_and_stop_is_not_a_failure(tmp_path: Path):
    assert _finish(tmp_path, _Result()) == []
    posts = _finish(tmp_path, _Result(is_error=True, terminal_reason="aborted_streaming"))
    assert not any(":x:" in p for p in posts)
