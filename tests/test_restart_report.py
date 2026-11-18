"""Messages accepted but not finished are reported after a bot restart (review R3); never replayed."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from unittest.mock import AsyncMock

from slack_agent_controller.hub import Hub
from slack_agent_controller.session import Prompt
from tests.helpers import FakeWeb, make_config


def _hub(tmp: Path) -> Hub:
    return Hub(make_config(tmp, [tmp]), FakeWeb())  # type: ignore[arg-type]


def _fake_turns(sess, gate: asyncio.Event) -> list[str]:  # noqa: ANN001
    ran: list[str] = []

    async def fake_turn(self, item, note):  # noqa: ANN001
        ran.append(item.text)
        await gate.wait()

    sess._run_turn = types.MethodType(fake_turn, sess)
    sess._reload_if_changed = AsyncMock(return_value=None)
    return ran


def test_records_follow_a_message_through_its_life(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        gate = asyncio.Event()
        _fake_turns(s, gate)
        s.submit("first job", "UOWNER")
        s.submit("second job", "UOWNER")
        await asyncio.sleep(0.1)
        statuses = sorted(r.status for r in hub.tasks.records.values())
        assert statuses == ["queued", "running"], statuses
        gate.set()
        for _ in range(50):
            if not hub.tasks.records:
                break
            await asyncio.sleep(0.02)
        assert not hub.tasks.records, "finished messages leave no record"
        await hub.shutdown()
    asyncio.run(run())


def test_restart_reports_interrupted_and_queued_but_runs_nothing(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        gate = asyncio.Event()
        _fake_turns(s, gate)
        s.submit("train the model", "UOWNER")
        s.submit("then plot the loss", "UOWNER")
        await asyncio.sleep(0.1)

        async def stop_turn(*a, **k):  # the shutdown interrupts the running reply
            gate.set()

        s.stop = stop_turn  # type: ignore[method-assign]
        s._turn = object()  # type: ignore[assignment]  # busy
        await hub.shutdown()
        recs = {r.text: r.status for r in hub.tasks.records.values()}
        assert recs == {"train the model": "interrupted", "then plot the loss": "queued"}, recs

        hub2 = _hub(tmp_path)                       # the restarted bot
        n = await hub2.report_unfinished()
        assert n == 2 and not hub2.tasks.records
        texts = hub2.web.texts()                    # type: ignore[attr-defined]
        assert len(texts) == 1, "one message per thread"
        t = texts[0]
        assert "<@UOWNER>" in t and "Interrupted while running" in t and "train the model" in t
        assert "Not started" in t and "then plot the loss" in t
        assert not hub2.sessions, "nothing was resent to Claude"
        assert await _hub(tmp_path).report_unfinished() == 0, "reported once"
    asyncio.run(run())


def test_crash_leaves_running_record_that_is_reported(tmp_path: Path):
    hub = _hub(tmp_path)
    tid = hub.tasks.add("D1", "1.0", "sid", "UOWNER", "long job")
    hub.tasks.mark(tid, "running")                  # then the process dies (no shutdown)

    async def run():
        hub2 = _hub(tmp_path)
        assert await hub2.report_unfinished() == 1
        assert "Interrupted while running" in hub2.web.texts()[0]  # type: ignore[attr-defined]
    asyncio.run(run())


def test_stop_and_terminal_refusal_clear_records(tmp_path: Path):
    async def run():
        hub = _hub(tmp_path)
        s = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        s.worker = asyncio.create_task(asyncio.sleep(5))   # keep the queue from being consumed
        s.submit("a", "UOWNER")
        s.submit("b", "UOWNER")
        assert len(hub.tasks.records) == 2
        await s.stop("UOWNER")                              # !stop drops the queue -> no stale records
        assert not hub.tasks.records
        s.worker.cancel()

        class Lock:
            pid, opened_at, background, session_id = 1, 0.0, False, s.rec.session_id

        class LockedMirror:
            def locked(self, sid):  # noqa: ANN001
                return Lock()

            def lock_text(self, st):  # noqa: ANN001
                return "locked"

        hub.mirror = LockedMirror()  # type: ignore[assignment]
        tid = hub.tasks.add("D1", "1.0", s.rec.session_id, "UOWNER", "c")
        await s._run_prompt(Prompt("c", "UOWNER", tid))
        assert not hub.tasks.records, "a refused message is not reported later"
    asyncio.run(run())
