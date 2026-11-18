"""Smoke test of SDK behaviours the bot relies on (no Slack involved).

Run:  .venv/bin/python tests/sdk_smoke.py
Uses the Haiku model on Bedrock; costs a few cents.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

for k in list(os.environ):
    if k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PID") or k.startswith(
        "CLAUDE_CODE_MESSAGING_"
    ):
        os.environ.pop(k)

hook_calls: list[tuple[str, str]] = []
perm_calls: list[str] = []


async def hook(inp, tool_use_id, ctx):
    tool = inp["tool_name"]
    decision = "allow" if tool in ("Read", "Glob", "Grep") else "ask"
    hook_calls.append((tool, decision))
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                   "permissionDecisionReason": "smoke"}}


async def can_use_tool(tool, inp, ctx):
    perm_calls.append(tool)
    return PermissionResultAllow(updated_input=inp)


async def main() -> None:
    work = tempfile.mkdtemp(prefix="sac_smoke_")
    sid = str(uuid.uuid4())
    opts = ClaudeAgentOptions(
        cwd=work,
        session_id=sid,
        model="haiku",
        permission_mode="default",
        setting_sources=[],
        can_use_tool=can_use_tool,
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[hook])]},
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "0", "DISABLE_AUTOUPDATER": "1"},
        stderr=lambda line: print("STDERR:", line[:200], file=sys.stderr),
    )

    async def turn(client: ClaudeSDKClient, prompt: str, label: str) -> ResultMessage | None:
        await client.query(prompt)
        res = None
        async for m in client.receive_response():
            if isinstance(m, SystemMessage) and m.subtype == "init":
                print(f"[{label}] init session_id={m.data.get('session_id')} model={m.data.get('model')} "
                      f"apiKeySource={m.data.get('apiKeySource')}")
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if isinstance(b, TextBlock):
                        print(f"[{label}] text: {b.text[:300]!r}")
                    elif isinstance(b, ToolUseBlock):
                        print(f"[{label}] tool_use: {b.name} {str(b.input)[:150]}")
            elif isinstance(m, UserMessage) and isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        print(f"[{label}] tool_result err={b.is_error}: {str(b.content)[:300]!r}")
            elif isinstance(m, ResultMessage):
                res = m
                print(f"[{label}] RESULT sid={m.session_id} cost={m.total_cost_usd} usage={m.usage} "
                      f"terminal={m.terminal_reason} subtype={m.subtype}")
        return res

    async with ClaudeSDKClient(opts) as client:
        await turn(client, "Use the Bash tool to run exactly: env | grep -c AWS_BEARER_TOKEN_BEDROCK; env | grep -c AWS_REGION ; "
                           "then reply with the raw output only.", "t1")
        await turn(client, "Reply with just the word OK.", "t2")

        # interrupt from another task
        async def stopper():
            await asyncio.sleep(15)
            print("[t3] calling interrupt() from another task")
            await client.interrupt()

        task = asyncio.create_task(stopper())
        await turn(client, 'Use Bash to run exactly: python3 -c "import time; '
                           '[print(i, flush=True) or time.sleep(1) for i in range(60)]"', "t3")
        await task
        await turn(client, "Reply with just the word AGAIN.", "t4")

    print("hook_calls:", hook_calls)
    print("perm_calls:", perm_calls)
    print("session transcript:", os.path.expanduser("~/.claude/projects"), sid)


asyncio.run(main())
