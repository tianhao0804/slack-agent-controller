"""End-to-end test of sessions/approvals against real Bedrock, fake Slack.

Run:  .venv/bin/python -m tests.integration_fake_slack
Uses the Haiku model; costs roughly $0.10-0.30.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from slack_agent_controller.config import strip_parent_session_env
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, is_card, is_reply, make_config

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"

mode = {"click": "approve"}  # what the simulated user does with the next approval
results: list[tuple[str, bool]] = []


def _remove_test_memory_dirs() -> None:
    """Test sessions run in /tmp/sac_* projects; Claude Code creates their (empty) auto-memory
    folders under ~/.claude/projects. Remove the empty ones so tests leave nothing behind."""
    import glob
    import os

    for d in glob.glob(os.path.expanduser("~/.claude/projects/-tmp-sac-*/memory")):
        try:
            os.rmdir(d)  # only succeeds when empty
        except OSError:
            pass


atexit.register(_remove_test_memory_dirs)


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)


async def clicker(hub: Hub) -> None:
    """Simulates the phone user: first an intruder click, then the owner."""
    handled: set[str] = set()
    while True:
        for aid in list(hub.approvals):
            if aid in handled:
                continue
            handled.add(aid)
            check("non-allowlisted click rejected", hub.resolve_approval(aid, "approve", "UEVIL") == "forbidden")
            if mode["click"] in ("approve", "deny", "approve_all"):
                await asyncio.sleep(0.5)
                hub.resolve_approval(aid, mode["click"], "UOWNER")  # type: ignore[arg-type]
        await asyncio.sleep(0.2)


async def wait_idle(sess, timeout: float = 240) -> None:
    t0 = time.monotonic()
    await asyncio.sleep(0.5)
    while sess.busy or not sess.queue.empty():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("turn did not finish")
        await asyncio.sleep(0.3)
    await asyncio.sleep(0.5)


def last_footer(web: FakeWeb) -> str:
    ups = [kw["text"] for op, kw in web.calls if op == "update" and is_reply(kw)]
    return ups[-1] if ups else ""


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_it_"))
    work = tmp / "proj"
    work.mkdir()
    secret = tmp / "notes"
    secret.mkdir()
    (secret / "k.txt").write_text("TOPSECRET")
    cfg = make_config(tmp, roots=[tmp], sensitive=[secret], model="haiku", approval_timeout_s=8.0,
                      default_cwd=work.resolve(), idle_disconnect_s=10.0,
                      read_only_roots=(Path(f"/tmp/claude-{os.getuid()}").resolve(),))
    web = FakeWeb()
    hub = Hub(cfg, web)  # type: ignore[arg-type]
    clk = asyncio.create_task(clicker(hub))

    sess = hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    sid = sess.rec.session_id

    # 1. Bash approval -> approve -> streamed output
    mode["click"] = "approve"
    sess.submit("Use the Bash tool to run `nvidia-smi -L | wc -l > gpu_count.txt && cat gpu_count.txt` and tell me the number of GPUs. "
                "The codeword for this chat is PINEAPPLE.", "UOWNER")
    await wait_idle(sess)
    cards = [kw for op, kw in web.calls if op == "post" and is_card(kw) and "Approval needed" in kw["text"]]
    check("approval card posted for Bash", len(cards) >= 1)
    check("approval card mentions owner", bool(cards) and "<@UOWNER>" in json.dumps(cards[0]["blocks"]))
    approved = [kw for op, kw in web.calls if op == "update" and "Approved" in kw.get("text", "")]
    check("approval card updated to Approved", len(approved) >= 1)
    check("answer contains 8", "8" in last_footer(web) or any("8" in t for t in web.texts()), last_footer(web)[-300:])
    check("streaming used chat.update", sum(1 for op, _ in web.calls if op == "update") >= 2)

    # 2. Context in the same thread
    sess.submit("What is the codeword for this chat? Answer with just the word.", "UOWNER")
    await wait_idle(sess)
    check("context kept within thread", "PINEAPPLE" in last_footer(web), last_footer(web)[-200:])

    # 3. Deny a Write
    mode["click"] = "deny"
    sess.submit("Use the Write tool to create hello.txt containing 'hi'. If denied, just say DENIED.", "UOWNER")
    await wait_idle(sess)
    check("denied Write did not create file", not (work / "hello.txt").exists())

    # 4. Timeout -> auto deny
    mode["click"] = "none"
    sess.submit("Use the Bash tool to run `touch timeout_marker`. If it is denied, reply TIMEOUT-DENIED and stop.",
                "UOWNER")
    await wait_idle(sess)
    check("approval timeout auto-denied", not (work / "timeout_marker").exists()
          and any("denied" in kw.get("text", "") and "No response" in kw.get("text", "")
                  for op, kw in web.calls if op == "update"))

    # 5. Protected path is denied by policy without asking
    n_cards = len([1 for op, kw in web.calls if op == "post" and is_card(kw)])
    direct = await sess._pre_tool_hook(
        {"tool_name": "Read", "tool_input": {"file_path": str(secret / "k.txt")}, "cwd": str(work)}, "x", None)
    check("hook denies protected Read (direct)",
          direct["hookSpecificOutput"]["permissionDecision"] == "deny")
    sess.submit(f"Use the Read tool to read the notes file {secret / 'k.txt'} and summarise it.", "UOWNER")
    await wait_idle(sess)
    audit = [json.loads(x) for x in (cfg.state_dir / "audit.jsonl").read_text().splitlines()]
    check("protected Read denied by policy", any(a.get("tool") == "Read" and a.get("decision") == "deny" for a in audit))
    check("protected Read not echoed", not any("TOPSECRET" in t for t in web.texts()))
    check("no approval card for protected Read",
          len([1 for op, kw in web.calls if op == "post" and is_card(kw)]) == n_cards)

    # 6. !stop during a long approved command
    mode["click"] = "approve"
    sess.submit('Use Bash to run exactly this command in the FOREGROUND (do not set run_in_background): '
                'python3 -c "import time; [print(i, flush=True) or time.sleep(1) for i in range(90)]"', "UOWNER")
    t0 = time.monotonic()
    while not sess.busy or not any("Approved" in kw.get("text", "") for op, kw in web.calls[-10:] if op == "update"):
        await asyncio.sleep(0.5)
        if time.monotonic() - t0 > 60:
            break
    await asyncio.sleep(6)
    stopped = await sess.stop("UOWNER")
    await wait_idle(sess)
    elapsed = time.monotonic() - t0
    check("!stop interrupted long task", stopped and elapsed < 60 and "interrupted" in last_footer(web),
          f"{elapsed:.0f}s {last_footer(web)[-120:]}")

    # 6b. Defects A+B: background task -> Claude-initiated turn is shown; next reply is not misaligned;
    #     the idle reaper (10s here) must not kill a process with a running background task.
    mode["click"] = "approve"
    sess.submit("Start this Bash command with run_in_background=true: `sleep 25 && echo BGDONE`. Then immediately "
                "end your turn by replying STARTED. When it finishes, report its output.", "UOWNER")
    await wait_idle(sess)
    check("background task tracked", len(sess.bg_tasks) == 1, str(sess.bg_tasks))
    await asyncio.sleep(14)
    check("not reaped while background task runs", sess.connected and sess.worker is not None
          and not sess.worker.done())
    t0 = time.monotonic()
    while time.monotonic() - t0 < 120 and not any(
            "Claude reported back" in kw.get("text", "") for op, kw in web.calls if op == "post"):
        await asyncio.sleep(1)
    await wait_idle(sess)
    check("Claude-initiated turn notified", any("Claude reported back" in kw.get("text", "")
                                                for op, kw in web.calls if op == "post"))
    check("Claude-initiated turn rendered", ":robot_face:" in last_footer(web) and "BGDONE" in last_footer(web),
          last_footer(web)[-300:])
    check("background task cleared", not sess.bg_tasks, str(sess.bg_tasks))
    sess.submit("Reply with just the word SECONDREPLY.", "UOWNER")
    await wait_idle(sess)
    check("no reply misalignment after Claude-initiated turn", "SECONDREPLY" in last_footer(web)
          and ":robot_face:" not in last_footer(web), last_footer(web)[-200:])
    await asyncio.sleep(14)
    check("idle session reaped when nothing runs", not sess.connected)

    # 7. Restart: new Hub from the same state dir resumes the thread
    await hub.shutdown()
    hub2 = Hub(cfg, web)  # type: ignore[arg-type]
    clk.cancel()
    clk = asyncio.create_task(clicker(hub2))
    s2 = hub2.get_session("D1", "1.0")
    check("mapping persisted", s2 is not None and s2.rec.session_id == sid and s2.rec.has_history)
    assert s2 is not None
    s2.submit("After the restart: what is the codeword for this chat? Just the word.", "UOWNER")
    await wait_idle(s2)
    check("context kept across bot restart", "PINEAPPLE" in last_footer(web), last_footer(web)[-200:])
    check("cost/tokens accumulated", s2.rec.cost_usd > 0 and s2.rec.turns >= 7, f"{s2.rec}")
    await hub2.shutdown()
    clk.cancel()

    print(f"\nsession {sid} cwd {work.resolve()}")
    print(f"cost ${s2.rec.cost_usd:.3f}, turns {s2.rec.turns}")
    print("audit log:", cfg.state_dir / "audit.jsonl")
    print(f"TERMINAL_RESUME cd {work.resolve()} && claude --resume {sid}")
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
