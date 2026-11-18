"""Integration test: ask-mode cards, built-in read-only commands and !mode auto|ask (real Bedrock, Sonnet; fake Slack).

Run:  .venv/bin/python -m tests.integration_mode
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from slack_agent_controller.config import strip_parent_session_env
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
mode = {"click": "approve"}


def cards(web: FakeWeb) -> int:
    return sum(1 for op, kw in web.calls if op == "post" and "Approval needed" in kw.get("text", ""))


async def clicker(hub: Hub) -> None:
    seen: set[str] = set()
    while True:
        for aid in list(hub.approvals):
            if aid not in seen:
                seen.add(aid)
                await asyncio.sleep(0.3)
                hub.resolve_approval(aid, mode["click"], "UOWNER")  # type: ignore[arg-type]
        await asyncio.sleep(0.2)


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_mode_"))
    work = tmp / "proj"
    work.mkdir()
    cfg = make_config(tmp, roots=[tmp], model="sonnet", default_cwd=work.resolve())
    web = FakeWeb()
    hub = Hub(cfg, web)  # type: ignore[arg-type]
    clk = asyncio.create_task(clicker(hub))
    sess = hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    check("new session defaults to ask mode", sess.rec.mode == "ask")

    n = cards(web)
    sess.submit("Run exactly this Bash command and report the line count: ls -la | wc -l", "UOWNER")
    await wait_idle(sess)
    check("ask mode: even a read-only Bash asks (no program-name auto-allow)", cards(web) == n + 1)

    n = cards(web)
    sess.submit("Run exactly this Bash command and report whether it succeeded: nvidia-smi -L", "UOWNER")
    await wait_idle(sess)
    check("built-in read-only command ran without a card", cards(web) == n)

    n = cards(web)
    sess.submit("Run exactly this Bash command: touch ask_marker.txt", "UOWNER")
    await wait_idle(sess)
    check("ask mode: side effect asked in Slack", cards(web) == n + 1 and (work / "ask_marker.txt").exists())

    await sess.set_mode("auto")  # live switch on a connected CLI
    n = cards(web)
    sess.submit("Run exactly this Bash command: touch auto_marker.txt", "UOWNER")
    await wait_idle(sess)
    check("auto mode: classifier approved without card", cards(web) == n and (work / "auto_marker.txt").exists())

    mode["click"] = "deny"
    n = cards(web)
    sess.submit("Run exactly this Bash command and report the number: printenv | wc -l. If denied, say DENIED.",
                "UOWNER")
    await wait_idle(sess)
    check("auto mode: env-dumping command still asks in Slack", cards(web) == n + 1)

    audit = [json.loads(x) for x in (cfg.state_dir / "audit.jsonl").read_text().splitlines()]
    check("audit records deferral to classifier", any(a.get("decision") == "deferred" for a in audit))

    # restart: mode persists and the CLI comes back in auto mode
    await hub.shutdown()
    hub2 = Hub(cfg, web)  # type: ignore[arg-type]
    s2 = hub2.get_session("D1", "1.0")
    assert s2 is not None
    check("mode persisted across restart", s2.rec.mode == "auto")
    s2.submit("Reply with just: OK", "UOWNER")
    await wait_idle(s2)
    downgraded = any("auto mode is not available" in kw.get("text", "") for op, kw in web.calls if op == "update")
    check("resumed CLI runs in auto mode", not downgraded)
    await hub2.shutdown()
    clk.cancel()

    bad = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    print(f"cost ${s2.rec.cost_usd:.3f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
