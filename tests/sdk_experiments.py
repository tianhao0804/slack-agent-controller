"""Experiments on SDK behaviour the bot must handle (real Bedrock, Haiku).

exp1: what arrives after a turn ends when a background Bash task finishes
      (Claude-initiated turn).
exp2: permission_mode="auto" vs PreToolUse hook vs can_use_tool.

Run: .venv/bin/python -m tests.sdk_experiments [exp1|exp2]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from slack_agent_controller.config import strip_parent_session_env

strip_parent_session_env()
T0 = time.monotonic()


def show(m) -> None:
    t = f"{time.monotonic() - T0:6.1f}s"
    if isinstance(m, StreamEvent) or (isinstance(m, SystemMessage) and m.subtype == "thinking_tokens"):
        return
    if isinstance(m, SystemMessage):
        keys = {k: m.data.get(k) for k in ("task_id", "status", "description", "summary", "tool_use_id") if k in m.data}
        print(t, f"System[{m.subtype}] {type(m).__name__} {keys}")
    elif isinstance(m, AssistantMessage):
        for b in m.content:
            if isinstance(b, TextBlock):
                print(t, "Assistant text:", b.text[:150].replace("\n", " | "))
            elif isinstance(b, ToolUseBlock):
                print(t, "Assistant tool_use:", b.name, str(b.input)[:120])
    elif isinstance(m, UserMessage):
        if isinstance(m.content, list):
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    print(t, f"ToolResult err={b.is_error}:", str(b.content)[:150].replace("\n", " | "))
        else:
            print(t, "User:", str(m.content)[:150].replace("\n", " | "), "origin=", m.origin)
    elif isinstance(m, ResultMessage):
        print(t, f"RESULT subtype={m.subtype} origin={m.origin} terminal={m.terminal_reason}")
    else:
        print(t, type(m).__name__)


async def drain(client: ClaudeSDKClient, seconds: float) -> None:
    """Read everything that arrives for `seconds` (simulates a continuous reader)."""
    async def reader():
        async for m in client.receive_messages():
            show(m)
    try:
        await asyncio.wait_for(reader(), timeout=seconds)
    except TimeoutError:
        pass


async def exp1() -> None:
    async def allow(tool, inp, ctx):
        return PermissionResultAllow(updated_input=inp)

    opts = ClaudeAgentOptions(cwd=tempfile.mkdtemp(), model="haiku", setting_sources=[], can_use_tool=allow,
                              env={"CLAUDE_CODE_USE_BEDROCK": "1"})
    async with ClaudeSDKClient(opts) as c:
        await c.query("Start this Bash command with run_in_background=true: `sleep 20 && echo BGDONE`. "
                      "Then immediately end your turn saying STARTED. When it finishes, report its output.")
        print("--- reading for 75s (turn + Claude-initiated follow-up)")
        await drain(c, 75)
        print("--- now a user prompt; is the next result ours?")
        await c.query("Reply with just: SECOND")
        async for m in c.receive_response():
            show(m)


async def exp2() -> None:
    perm: list[str] = []

    async def hook(inp, tid, ctx):
        tool = inp["tool_name"]
        cmd = str(inp["tool_input"].get("command", ""))
        if "FORCEASK" in cmd:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                           "permissionDecisionReason": "test ask"}}
        print(f"   hook: {tool} -> no decision")
        return {}

    async def can_use(tool, inp, ctx):
        perm.append(f"{tool}:{inp.get('command', '')[:60]}")
        print(f"   can_use_tool called: {tool} {inp.get('command', '')[:80]!r} reason={ctx.decision_reason!r}")
        return PermissionResultDeny(message="denied by experiment")

    work = tempfile.mkdtemp()
    opts = ClaudeAgentOptions(cwd=work, model="sonnet", setting_sources=[], permission_mode="auto",
                              can_use_tool=can_use, hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[hook])]},
                              env={"CLAUDE_CODE_USE_BEDROCK": "1"},
                              stderr=lambda s: print("   STDERR", s[:160]) if s.strip() else None)
    prompts = [
        ("benign", "Run this exact Bash command and report the output: ls -la"),
        ("side-effect", "Run this exact Bash command and report the result: touch auto_mode_marker.txt && ls"),
        ("pip", "Run this exact Bash command once and report the result: pip install --user --dry-run requests"),
        ("forced-ask", "Run this exact Bash command: echo FORCEASK. If blocked, say BLOCKED and stop."),
    ]
    async with ClaudeSDKClient(opts) as c:
        for label, p in prompts:
            print(f"=== {label}")
            await c.query(p)
            async for m in c.receive_response():
                show(m)
    print("can_use_tool calls:", perm)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "exp1"
    asyncio.run({"exp1": exp1, "exp2": exp2}[which]())
