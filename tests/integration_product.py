"""Integration (real Bedrock/Haiku, fake Slack): Claude's question answered by a button and by a
thread reply; "Interrupt and send now" on a queued message.

Run: .venv/bin/python -m tests.integration_product
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.config import strip_parent_session_env
from tests.helpers import FakeWeb, is_reply, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
ASK = ("Before anything else, use the AskUserQuestion tool to ask me which color I prefer, with exactly two "
       "options: Red and Blue. After I answer, reply with just the color I gave, in capital letters.")


def last_reply(web: FakeWeb) -> str:
    ups = [kw["text"] for op, kw in web.calls if op == "update" and is_reply(kw)]
    return ups[-1] if ups else ""


async def wait_for(cond, timeout: float = 120) -> bool:  # noqa: ANN001
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.3)
    return False


async def approver(f: SlackFrontend) -> None:
    while True:
        for aid in list(f.hub.approvals):
            f.hub.resolve_approval(aid, "approve", "UOWNER")
        await asyncio.sleep(0.3)


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_prod_"))
    work = tmp / "proj"
    work.mkdir()
    f = SlackFrontend(make_config(tmp, roots=[tmp], model="haiku", default_cwd=work.resolve()))
    web = FakeWeb()
    f.hub.web = web  # type: ignore[assignment]
    clk = asyncio.create_task(approver(f))

    # 1) question answered with a button
    s = f.hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    s.submit(ASK, "UOWNER")
    got = await wait_for(lambda: bool(f.hub.questions))
    check("question card shown", got)
    if got:
        q = next(iter(f.hub.questions.values()))
        labels = q.labels(0)
        check("options are Red and Blue", sorted(labels) == ["Blue", "Red"], str(labels))
        oi = labels.index("Blue") if "Blue" in labels else 0
        await f._on_question({"user": {"id": "UOWNER"}, "channel": {"id": "D1"}, "message": {"ts": q.ts}},
                             {"action_id": f"sac_qa_pick_0_{oi}", "value": f"{q.id}|0|{oi}"})
    await wait_idle(s, 180)
    check("Claude continued with the button answer", "BLUE" in last_reply(web), last_reply(web)[-200:])

    # 2) question answered with a thread reply
    s2 = f.hub.create_session("D1", "2.0", work.resolve(), "UOWNER")
    s2.submit(ASK, "UOWNER")
    got = await wait_for(lambda: bool(f.hub.questions))
    check("second question card shown", got)
    if got:
        await f._prompt("D1", "2.5", "2.0", "UOWNER", "green")
    await wait_idle(s2, 180)
    check("Claude continued with the typed answer", "GREEN" in last_reply(web), last_reply(web)[-200:])
    audit = (f.cfg.state_dir / "audit.jsonl").read_text()
    check("questions audited as answered", audit.count('"event": "question"') >= 2 and '"outcome": "answers"' in audit)

    # 3) "Interrupt and send now" on a queued message
    s3 = f.hub.create_session("D1", "3.0", work.resolve(), "UOWNER")
    s3.submit('Run this Bash command in the FOREGROUND (do not set run_in_background) and then reply DONE: '
              'python3 -c "import time; time.sleep(90)"', "UOWNER")
    running = await wait_for(lambda: s3.busy and any("sleep(90)" in t for t in s3.open_tools.values()), 120)
    check("long command running in the foreground", running and not s3.bg_tasks,
          f"busy={s3.busy} open={list(s3.open_tools.values())} bg={list(s3.bg_tasks.values())}")
    await asyncio.sleep(2)
    t0 = time.monotonic()
    await f._prompt("D1", "3.5", "3.0", "UOWNER", "Forget that. Reply with just the word: SECOND")
    notice = [kw for op, kw in web.calls if op == "post" and kw.get("blocks")
              and any(b.get("block_id") == "sac_queue" for b in kw["blocks"])]
    check("queue notice with buttons", bool(notice))
    if notice:
        btn = next(b for b in notice[-1]["blocks"] if b.get("block_id") == "sac_queue")["elements"][0]
        await f._on_queue_choice({"user": {"id": "UOWNER"}, "channel": {"id": "D1"}, "message": {"ts": "x"}}, btn)
    await wait_idle(s3, 180)
    took = time.monotonic() - t0
    check("queued message answered right away (not after the 90 s command)", "SECOND" in last_reply(web) and took < 80,
          f"{took:.0f}s {last_reply(web)[-120:]}")
    tasks = json.loads((f.cfg.state_dir / "tasks.json").read_text())["tasks"]
    check("no task records left behind", tasks == [], str(tasks))

    clk.cancel()
    await f.hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
