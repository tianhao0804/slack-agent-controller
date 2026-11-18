"""Experiment: how does a session's transcript change after (a) the bot's own turn
ends and (b) a terminal `claude --resume` adds a turn? (real Bedrock, Haiku)

Run: .venv/bin/python -m tests.exp_transcript_changes
"""

from __future__ import annotations

import asyncio
import glob
import os
import subprocess
import tempfile
import time
import uuid

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, get_session_info

from slack_agent_controller.config import strip_parent_session_env

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"


def fp(sid: str) -> tuple[int | None, int | None]:
    info = get_session_info(sid)
    return (info.file_size, info.last_modified) if info else (None, None)


def files_in_project(cwd: str) -> list[str]:
    slug = os.path.realpath(cwd).replace("/", "-").replace("_", "-")
    return sorted(os.path.basename(p) for p in glob.glob(os.path.expanduser(f"~/.claude/projects/{slug}/*.jsonl")))


async def main() -> None:
    cwd = tempfile.mkdtemp(prefix="sac_exp_")
    sid = str(uuid.uuid4())
    opts = ClaudeAgentOptions(cwd=cwd, session_id=sid, model="haiku", setting_sources=[],
                              env={"CLAUDE_CODE_USE_BEDROCK": "1"})
    async with ClaudeSDKClient(opts) as c:
        await c.query("The codeword for this chat is KIWI. Reply OK.")
        async for _ in c.receive_response():
            pass
        base = fp(sid)
        print("at result       :", base)
        t0 = time.monotonic()
        changes = []
        while time.monotonic() - t0 < 12:
            await asyncio.sleep(0.5)
            cur = fp(sid)
            if cur != base:
                changes.append((round(time.monotonic() - t0, 1), cur))
                base = cur
        print("changes after result while connected (t, (size, mtime)):", changes or "none")

        before = fp(sid)
        env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
        out = subprocess.run(
            ["claude", "-p", "--model", "haiku", "--resume", sid, "A second codeword for this chat is PEAR. Reply OK."],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        print("terminal rc/out :", out.returncode, out.stdout.strip()[:60])
        after = fp(sid)
        print("size before/after terminal turn:", before[0], "->", after[0], "| grew:", (after[0] or 0) > (before[0] or 0))
        print("jsonl files in project dir:", files_in_project(cwd))
        print("original id still in dir  :", f"{sid}.jsonl" in files_in_project(cwd))


asyncio.run(main())
