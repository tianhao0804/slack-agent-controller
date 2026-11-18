"""Integration: terminal `claude -p` with the mirror hooks -> Slack thread (real Bedrock/Haiku, fake Slack).

The hooks are passed with --settings, so ~/.claude/settings.json is not touched.
Run: .venv/bin/python -m tests.integration_mirror
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.config import strip_parent_session_env
from slack_agent_controller.mirror import Mirror
from tests.helpers import FakeWeb, is_reply, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
ROOT = Path(__file__).resolve().parent.parent


def terminal(cwd: Path, settings: Path, prompt: str, *extra: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["claude", "-p", "--model", "haiku", "--settings", str(settings), "--output-format", "json", *extra, prompt],
        cwd=cwd, env={**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)


async def finish(p: subprocess.Popen[str]) -> dict:
    out, _ = await asyncio.to_thread(p.communicate, timeout=240)
    try:
        return json.loads(out)
    except ValueError:
        return {"result": out}


async def until(cond, timeout: float = 40.0) -> bool:  # noqa: ANN001
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.2)
    return False


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_mir_"))
    proj = tmp / "proj"
    proj.mkdir()
    (proj / "notes.txt").write_text("The codeword is MANGO.\n")
    (proj / ".env").write_text("LAB_NOTE=SECRETVALUE123\n")
    sock = tmp / "m.sock"
    cfg = make_config(tmp, roots=[tmp], model="haiku", allowed_channels=frozenset({"CMIR"}),
                      mirror_dirs=(proj.resolve(),), mirror_channel="CMIR", mirror_user="UOWNER", mirror_socket=sock)
    front = SlackFrontend(cfg)
    web = FakeWeb()
    front.hub.web = web  # type: ignore[assignment]
    hub = front.hub
    mirror = Mirror(hub)
    hub.mirror = mirror
    await mirror.start()

    hook = {"type": "command", "command": f"{ROOT}/.venv/bin/python -I {ROOT}/deploy/sac_hook.py --socket {sock}",
            "timeout": 5}
    hooks = {e: [{"hooks": [hook]}] for e in ("SessionStart", "UserPromptSubmit", "Stop", "Notification", "SessionEnd")}
    hooks["PreToolUse"] = [{"matcher": "*", "hooks": [hook]}]
    settings = tmp / "settings.json"
    settings.write_text(json.dumps({"hooks": hooks}))
    uploads = lambda: [kw for op, kw in web.calls if op == "upload"]  # noqa: E731

    # 1) A terminal turn is mirrored into a new thread.
    out = await finish(terminal(proj, settings, "Run `cat notes.txt` and then `cat .env` as two separate Bash "
                                "commands, then reply with just the codeword from notes.txt.",
                                "--allowedTools=Bash(cat:*)"))
    sid = out.get("session_id", "")
    check("terminal turn ran", "MANGO" in str(out.get("result", "")), str(out)[:200])
    ok = await until(lambda: (st := mirror.states.get(sid)) is not None and st.turns == 1 and not st.open)
    check("turn mirrored and lock released at exit", ok)
    st = mirror.states.get(sid)
    rec = hub.store.find_by_session(sid)
    check("thread created in the mirror channel", rec is not None and rec.channel == "CMIR" and st is not None
          and st.thread_ts == rec.thread_ts)
    check("terminal pid recorded", st is not None and st.pid is not None)
    ups = uploads()
    md = ups[0]["content"] if ups else ""
    check("turn-001.md attached in the thread", len(ups) == 1 and ups[0]["filename"] == "turn-001.md"
          and rec is not None and ups[0]["thread_ts"] == rec.thread_ts)
    check("log has the command and its output", "cat notes.txt" in md and "The codeword is MANGO" in md, md[:300])
    check("protected file content omitted", "SECRETVALUE123" not in md
          and ("cat .env" not in md or "omitted: touches a protected path" in md))
    texts = web.texts()
    check("progress showed tool calls", any(":hammer_and_wrench:" in kw.get("text", "") and "cat" in kw.get("text", "")
                                            for op, kw in web.calls if op in ("post", "update")))
    check("summary has Claude's reply", any("Terminal turn 1" in t and "MANGO" in t for t in texts))
    check("root shows terminal closed", rec is not None and "Terminal closed" in web.messages[rec.thread_ts]["text"])
    assert rec is not None and st is not None

    # 2) While a terminal has the session open, Slack prompts are refused.
    p = terminal(proj, settings, "Run this Bash command: python3 -c \"import time; time.sleep(12)\" and then "
                 "reply with just: SLEPT", "--resume", sid, "--allowedTools=Bash(python3:*)")
    locked = await until(lambda: mirror.locked(sid) is not None, 60)
    check("lock taken when the terminal resumed", locked)
    check("lock notice posted in the thread", any("Opened in a terminal" in t for t in web.texts()))
    n_before = len(web.calls)
    await front._prompt(rec.channel, "99.0", rec.thread_ts, "UOWNER", "Slack message during terminal use")
    sess = hub.sessions.get(rec.key)
    check("Slack prompt refused while locked", any("open in a terminal" in kw.get("text", "")
                                                   for _, kw in web.calls[n_before:])
          and (sess is None or sess.worker is None))
    out2 = await finish(p)
    ok = await until(lambda: mirror.states[sid].turns == 2 and not mirror.states[sid].open)
    md2 = uploads()[-1]["content"] if len(uploads()) == 2 else ""
    check("second turn mirrored", ok and "SLEPT" in md2, str(out2)[:200])
    check("resumed turn's log has no earlier history", "cat notes.txt" not in md2)

    # 3) After the terminal exits, Slack continues the same session.
    await front._prompt(rec.channel, "100.0", rec.thread_ts, "UOWNER",
                        "What codeword did notes.txt contain? Reply with just the word.")
    sess = hub.sessions[rec.key]
    await wait_idle(sess)
    last = [kw["text"] for op, kw in web.calls if op == "update" and is_reply(kw) and
            kw.get("ts") not in (rec.thread_ts,)][-1]
    check("Slack turn continues the terminal session", "MANGO" in last, last[-200:])
    check("Slack turn used the same session id", sess.rec.session_id == sid)

    # 4) Terminal resumes again: bot releases its idle process; the log skips the Slack turn.
    connected_before = sess.connected
    out3 = await finish(terminal(proj, settings, "Reply with just: THIRD", "--resume", sid))
    ok = await until(lambda: mirror.states[sid].turns == 3 and not mirror.states[sid].open)
    md3 = uploads()[-1]["content"] if len(uploads()) == 3 else ""
    check("third turn mirrored", ok and "THIRD" in md3, str(out3)[:200])
    check("Slack's own turn not re-mirrored", "What codeword did notes.txt contain" not in md3)
    check("bot released its idle CLI process", connected_before and not sess.connected)

    await mirror.stop()
    await hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
