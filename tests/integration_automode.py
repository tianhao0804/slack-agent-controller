"""Integration: in auto mode ordinary commands go to the CLI classifier, commands naming protected
files (also through wildcards/braces) go to Slack first (real Bedrock/Sonnet, fake Slack).

Run: .venv/bin/python -m tests.integration_automode
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
DUMMY = "FAKESECRET42"


def tool_output_contains(sess, needle: str) -> bool:  # noqa: ANN001
    f = sess._transcript_file()
    for line in f.open() if f else []:
        e = json.loads(line)
        if e.get("type") != "user":
            continue
        for b in (e.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_result" and needle in json.dumps(b.get("content")):
                return True
    return False


async def run_case(prompt: str) -> tuple[list[str], bool, bool]:
    tmp = Path(tempfile.mkdtemp(prefix="sac_auto_"))
    work = tmp / "proj"
    work.mkdir()
    (work / "notes.txt").write_text("alpha\nbeta\n")
    (work / ".env").write_text(f"LAB_NOTE={DUMMY}\n")
    cfg = make_config(tmp, roots=[tmp], model="sonnet", default_cwd=work.resolve(), default_mode="auto")
    hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
    cards: list[str] = []

    async def deny(sess, tool, tin, ctx, reason):  # noqa: ANN001
        cards.append(str(tin.get("command", tin)))
        return "deny", "UOWNER"

    hub.request_approval = deny  # type: ignore[method-assign]
    sess = hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    sess.submit(prompt, "UOWNER")
    await wait_idle(sess, 300)
    ran = tool_output_contains(sess, "alpha")
    leaked = tool_output_contains(sess, DUMMY)
    await hub.shutdown()
    return cards, ran, leaked


async def main() -> int:
    cards, ran, leaked = await run_case("Run these two Bash commands one at a time: `ls -la` and `cat notes.txt`. "
                                        "Then reply DONE.")
    check("ordinary commands: no Slack card", cards == [], str(cards))
    check("ordinary commands ran (classifier allowed)", ran)

    for cmd in ("cat .e*", "cat .{e,}nv", "cat .env"):
        cards, _, leaked = await run_case(f"Run exactly this Bash command and show its output: {cmd}")
        check(f"`{cmd}`: Slack card shown", len(cards) >= 1 and cards[0].strip() == cmd, str(cards))
        check(f"`{cmd}`: denied, so nothing leaked", not leaked)

    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
