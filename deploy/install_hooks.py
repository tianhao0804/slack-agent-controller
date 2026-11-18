#!/usr/bin/env python3
"""Install / remove the terminal-mirror hooks in ~/.claude/settings.json.

    .venv/bin/python deploy/install_hooks.py            # install (or update)
    .venv/bin/python deploy/install_hooks.py --remove   # uninstall
    .venv/bin/python deploy/install_hooks.py --dry-run  # show the result only

Only hook entries whose command runs ``sac_hook.py`` are added or removed;
everything else in the file is left as is. A timestamped backup is written
next to the file before every change. Which sessions are mirrored is decided
by the bot (MIRROR_DIRS), so the hooks can be global. Running terminal
sessions pick up the hooks only after they are restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "sac_hook.py"
_VENV_PYTHON = HERE.parent / ".venv" / "bin" / "python"
# The hook script is standard-library only: the repository's venv if there is one, else this interpreter.
PYTHON = _VENV_PYTHON if os.access(_VENV_PYTHON, os.X_OK) else Path(sys.executable)
SETTINGS = Path.home() / ".claude" / "settings.json"
EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "Stop", "Notification", "SessionEnd")
MARK = "sac_hook.py"


def _ours(group: dict) -> bool:
    return any(MARK in str(h.get("command", "")) for h in group.get("hooks", []) if isinstance(h, dict))


def strip(settings: dict) -> dict:
    hooks = settings.get("hooks") or {}
    for ev in list(hooks):
        groups = [g for g in hooks[ev] if not (isinstance(g, dict) and _ours(g))]
        if groups:
            hooks[ev] = groups
        else:
            del hooks[ev]
    if hooks:
        settings["hooks"] = hooks
    else:
        settings.pop("hooks", None)
    return settings


def install(settings: dict, socket_path: str | None, python: Path = PYTHON) -> dict:
    settings = strip(settings)
    cmd = f"{python} -I {SCRIPT}" + (f" --socket {socket_path}" if socket_path else "")
    hook = {"type": "command", "command": cmd, "timeout": 5}
    hooks = settings.setdefault("hooks", {})
    for ev in EVENTS:
        group: dict = {"hooks": [dict(hook)]}
        if ev == "PreToolUse":
            group["matcher"] = "*"
        hooks.setdefault(ev, []).append(group)
    return settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--socket", help="socket path (default: the bot's default, /run/user/<uid>/sac-mirror.sock)")
    ap.add_argument("--settings", type=Path, default=SETTINGS)
    ap.add_argument("--python", type=Path, default=None, help=f"interpreter for the hook (default: {PYTHON})")
    a = ap.parse_args()
    python = a.python or PYTHON
    if not a.remove and not (SCRIPT.is_file() and os.access(python, os.X_OK)):
        print(f"missing {SCRIPT} or {python}", file=sys.stderr)
        return 1
    current = json.loads(a.settings.read_text()) if a.settings.exists() else {}
    new = strip(current) if a.remove else install(current, a.socket, python)
    text = json.dumps(new, indent=2, ensure_ascii=False) + "\n"
    if a.dry_run:
        print(text)
        return 0
    if a.settings.exists():
        backup = a.settings.with_name(f"{a.settings.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(a.settings, backup)
        print(f"backup: {backup}")
    tmp = a.settings.with_name(a.settings.name + ".tmp")
    tmp.write_text(text)
    os.chmod(tmp, a.settings.stat().st_mode & 0o777 if a.settings.exists() else 0o600)
    os.replace(tmp, a.settings)
    print(("removed mirror hooks from " if a.remove else "installed mirror hooks in ") + str(a.settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
