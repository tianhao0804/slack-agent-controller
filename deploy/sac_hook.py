#!/usr/bin/env python3
"""Claude Code hook -> slack-agent-controller terminal mirror (stdlib only, Python 3.8+).

Installed into ~/.claude/settings.json by deploy/install_hooks.py. Claude Code
runs it on SessionStart / UserPromptSubmit / PreToolUse / Stop / Notification /
SessionEnd with the hook payload on stdin. It forwards a trimmed copy of the
payload (no tool outputs) to the bot over a user-private Unix socket.

It must never get in the way of the terminal: it prints nothing (stdout of a
UserPromptSubmit/SessionStart hook would be added to Claude's context), always
exits 0 (exit 2 would block the action), and gives up within ~1 s if the bot
is not running.
"""

import json
import os
import socket
import sys

MAX_STR = 4000
KEEP = ("hook_event_name", "session_id", "transcript_path", "cwd", "source", "reason", "prompt",
        "tool_name", "tool_input", "tool_use_id", "message", "notification_type", "title", "agent_id",
        "last_assistant_message")


def _trim(v, depth=0):
    if isinstance(v, str):
        return v if len(v) <= MAX_STR else v[:MAX_STR] + "…"
    if isinstance(v, dict) and depth < 4:
        return {str(k): _trim(x, depth + 1) for k, x in list(v.items())[:50]}
    if isinstance(v, list) and depth < 4:
        return [_trim(x, depth + 1) for x in v[:50]]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)[:200]


def _start_time(pid):
    """Process start time (clock ticks since boot): pid + start time identifies a process."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except Exception:
        return None


def main():
    sock_path = sys.argv[sys.argv.index("--socket") + 1] if "--socket" in sys.argv else None
    if not sock_path:
        sock_path = "/run/user/%d/sac-mirror.sock" % os.getuid()
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    ev = {k: _trim(payload[k]) for k in KEEP if k in payload}
    try:
        pid = int(os.environ.get("CLAUDE_PID") or 0)
    except ValueError:
        pid = 0
    if pid:
        ev["claude_pid"] = pid
        ev["claude_pid_start"] = _start_time(pid)
    data = (json.dumps(ev, ensure_ascii=False) + "\n").encode()
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(sock_path)
        s.sendall(data)
        s.close()
    except Exception:
        pass  # bot not running: mirroring is best-effort


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
