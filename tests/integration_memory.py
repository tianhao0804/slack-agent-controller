"""Integration: auto-memory goes to the project's memory dir without approval cards, and a later
session in the same project sees it (real Bedrock/Sonnet, fake Slack, temporary project dir).

Run: .venv/bin/python -m tests.integration_memory
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

from slack_agent_controller import policy
from slack_agent_controller.config import strip_parent_session_env
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, is_reply, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_mem_"))
    proj = tmp / "proj"
    proj.mkdir()
    mem = policy.project_memory_dir(proj)
    cfg = make_config(tmp, roots=[tmp], model="sonnet", default_cwd=proj,
                      sensitive=[Path.home() / ".claude", Path.home() / ".ssh"])
    hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
    cards: list[str] = []

    async def deny(sess, tool, tin, ctx, reason):  # noqa: ANN001
        cards.append(f"{tool}: {tin.get('file_path') or tin.get('command')}")
        return "deny", "UOWNER"

    hub.request_approval = deny  # type: ignore[method-assign]
    try:
        s = hub.create_session("D1", "1.0", proj, "UOWNER")
        s.submit("Please save to your memory, for future sessions: my preferred functional is r2SCAN. "
                 "Then reply SAVED.", "UOWNER")
        await wait_idle(s, 240)
        files = sorted(p.name for p in mem.glob("*.md")) if mem.exists() else []
        text = " ".join(p.read_text() for p in mem.glob("*.md")) if mem.exists() else ""
        check("memory written to the project's memory dir", "r2SCAN" in text, f"{mem}: {files}")
        check("no approval card for memory", not cards, str(cards))

        s2 = hub.create_session("D1", "2.0", proj, "UOWNER")
        s2.submit("Which DFT functional do I prefer? Answer with just its name.", "UOWNER")
        await wait_idle(s2, 240)
        ans = [kw["text"] for op, kw in hub.web.calls if op == "update" and is_reply(kw)][-1]
        check("a new session in the same project recalls it", "r2SCAN" in ans, ans[-200:])
        await hub.shutdown()
    finally:
        if mem.parent.exists() and mem.parent.name.startswith("-tmp-sac-mem-"):
            shutil.rmtree(mem.parent)   # do not leave test notes in ~/.claude
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
