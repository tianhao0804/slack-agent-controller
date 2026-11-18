"""Integration test: !compact (real Bedrock, Haiku; fake Slack).

Run:  .venv/bin/python -m tests.integration_compact
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.config import strip_parent_session_env
from tests.helpers import FakeWeb, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
os.environ.setdefault("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")


def texts(web: FakeWeb) -> str:
    return "\n".join(str(kw.get("text", "")) for _, kw in web.calls)


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_compact_"))
    work = tmp / "proj"
    work.mkdir()
    f = SlackFrontend(make_config(tmp, roots=[tmp], model="sonnet", default_cwd=work.resolve()))
    web = FakeWeb()
    f.hub.web = web  # type: ignore[assignment]

    # outside a session thread: refused, nothing sent to Claude
    await f._cmd_compact("D1", "9.0", None, "UOWNER", "")
    check("refused outside a session thread", "inside a session thread" in texts(web))

    s = f.hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    s.submit("Remember this code word: PINEAPPLE-42. Reply with just: noted", "UOWNER")
    await wait_idle(s, 180)
    await f._cmd_compact("D1", "1.5", "1.0", "UOWNER", "keep the code word")
    await wait_idle(s, 240)
    check("compaction reported in the thread", "context compacted" in texts(web), texts(web)[-300:])
    check("the thread is idle and the session still bound", not s.busy and f.hub.store.get(s.rec.key) is not None)

    n = len(web.calls)
    await f._prompt("D1", "1.7", "1.0", "UOWNER", "What was the code word? Reply with just the word.")
    await wait_idle(s, 180)
    after = "\n".join(str(kw.get("text", "")) for _, kw in web.calls[n:])
    check("after compaction the conversation continues and remembers", "PINEAPPLE-42" in after, after[-200:])

    await f.hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
