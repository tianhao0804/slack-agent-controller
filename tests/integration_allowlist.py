"""Integration: an approved Bash command is remembered; destructive ones keep asking
(real Bedrock/Haiku, fake Slack; the approval click is simulated).

Run: .venv/bin/python -m tests.integration_allowlist
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


def executed_count(sess, needle: str) -> int:  # noqa: ANN001
    """Tool results containing ``needle`` in the session transcript (proves the command ran)."""
    f = sess._transcript_file()
    n = 0
    for line in f.open() if f else []:
        e = json.loads(line)
        if e.get("type") != "user":  # api-request-blob entries repeat earlier tool results
            continue
        for b in (e.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_result" and needle in json.dumps(b.get("content")):
                n += 1
    return n


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_allow_"))
    work = tmp / "proj"
    work.mkdir()
    cfg = make_config(tmp, roots=[tmp], model="haiku", default_cwd=work.resolve())
    hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
    cards: list[str] = []

    async def approve(sess, tool, tin, ctx, reason):  # noqa: ANN001
        cards.append(str(tin.get("command", "")))
        return ("remember" if hub.allowlist.can_remember(tin, sess.cwd) else "approve"), "UOWNER"

    hub.request_approval = approve  # type: ignore[method-assign]
    sess = hub.create_session("D1", "1.0", work.resolve(), "UOWNER")

    def ask(cmd: str) -> None:
        sess.submit(f"Run exactly this Bash command and nothing else, then reply DONE: {cmd}", "UOWNER")

    ask("echo ALLOWLIST_OK")
    await wait_idle(sess)
    check("first run asked once", cards == ["echo ALLOWLIST_OK"], str(cards))
    check("command remembered", len(hub.allowlist.entries) == 1)

    cards.clear()
    ask("echo ALLOWLIST_OK")
    await wait_idle(sess)
    audit = (cfg.state_dir / "audit.jsonl").read_text()
    check("second run did not ask", cards == [], str(cards))
    check("second run audited as allowlist", '"by": "allowlist"' in audit)
    check("command output present for both runs", executed_count(sess, "ALLOWLIST_OK") == 2,
          str(executed_count(sess, "ALLOWLIST_OK")))

    cards.clear()
    for _ in range(2):
        ask("rm -f nothing_here.txt")
        await wait_idle(sess)
    check("destructive command asked both times", len(cards) == 2, str(cards))
    check("destructive command not remembered", len(hub.allowlist.entries) == 1)

    await hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
