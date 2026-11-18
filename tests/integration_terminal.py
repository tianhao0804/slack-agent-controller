"""Integration: Slack turn -> terminal `claude --resume` turn -> Slack turn must see the terminal's work.
Real Bedrock (Haiku), fake Slack.

Run: .venv/bin/python -m tests.integration_terminal
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from slack_agent_controller.config import strip_parent_session_env
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, is_reply, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"


def last_update(web: FakeWeb) -> str:
    return [kw["text"] for op, kw in web.calls if op == "update" and is_reply(kw)][-1]


def reload_notes(web: FakeWeb) -> int:
    """Number of distinct Slack messages that carry the reload note."""
    return sum("outside Slack" in m.get("text", "") for m in web.messages.values())


async def terminal_turn(cwd: Path, sid: str, prompt: str) -> str:
    out = await asyncio.to_thread(
        subprocess.run, ["claude", "-p", "--model", "haiku", "--resume", sid, prompt],
        cwd=cwd, env={**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}, capture_output=True, text=True, timeout=180)
    return out.stdout.strip()


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_term_"))
    work = tmp / "proj"
    work.mkdir()
    cfg = make_config(tmp, roots=[tmp], model="haiku", default_cwd=work.resolve())
    web = FakeWeb()
    hub = Hub(cfg, web)  # type: ignore[arg-type]
    sess = hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    sid = sess.rec.session_id

    sess.submit("The codeword for this chat is KIWI. Reply OK.", "UOWNER")
    await wait_idle(sess)
    check("baseline recorded after Slack turn", sess._baseline is not None and sess._baseline > 0)
    first_client = sess.client

    # Control: no outside activity -> same process, no reload note.
    sess.submit("Reply with just: STILL-HERE", "UOWNER")
    await wait_idle(sess)
    check("no outside activity: process kept", sess.client is first_client and reload_notes(web) == 0)

    # The user continues the same session in a terminal.
    reply = await terminal_turn(work.resolve(), sid, "A second codeword for this chat is PEAR. Reply OK.")
    check("terminal turn ran", "OK" in reply, reply[:80])

    # Back in Slack while the bot's CLI process is still alive (and stale).
    check("bot process still connected (stale)", sess.client is first_client)
    sess.submit("List every codeword mentioned in this chat, comma separated.", "UOWNER")
    await wait_idle(sess)
    ans = last_update(web)
    check("reload note shown", reload_notes(web) == 1)
    check("process was replaced", sess.client is not first_client and sess.client is not None)
    check("Slack turn sees terminal's work", "KIWI" in ans and "PEAR" in ans, ans[-200:])

    # And it keeps going normally afterwards.
    again = sess.client
    sess.submit("Reply with just: AGAIN", "UOWNER")
    await wait_idle(sess)
    check("no further reload afterwards", sess.client is again and reload_notes(web) == 1)

    await hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    print(f"cost ${sess.rec.cost_usd:.3f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
