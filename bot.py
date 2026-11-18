#!/usr/bin/env python3
"""Entry point: ``python bot.py [--env-file PATH] [--check]``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from slack_agent_controller.config import ConfigError, load_config


def setup_logging(state_dir: Path, debug: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    stream = logging.StreamHandler(sys.stderr)  # -> journald under systemd
    stream.setFormatter(fmt)
    fileh = logging.handlers.RotatingFileHandler(state_dir / "bot.log", maxBytes=10_000_000, backupCount=5)
    fileh.setFormatter(fmt)
    root.addHandler(stream)
    root.addHandler(fileh)
    (state_dir / "bot.log").chmod(0o600)
    # Third-party loggers can be chatty (and slack_sdk DEBUG would include payloads).
    for name in ("slack_bolt", "slack_sdk", "asyncio", "aiohttp"):
        logging.getLogger(name).setLevel(logging.WARNING)


def acquire_instance_lock(state_dir: Path, lock_dir: Path | None = None) -> object | None:
    """Exclusive lock so only one bot runs per user and node (two copies would each
    keep their own state and overwrite each other's files). Node-local when possible."""
    import fcntl

    run = Path(f"/run/user/{os.getuid()}")
    path = (lock_dir or (run if run.is_dir() else state_dir)) / "slack-agent-controller.lock"
    f = open(path, "a+")  # noqa: SIM115 - held for the life of the process
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        other = f.read().strip() or "?"
        f.close()
        print(f"another slack-agent-controller is already running (pid {other}; lock {path}). Stop it first, e.g. "
              "`systemctl --user stop slack-agent-controller`.", file=sys.stderr)
        return None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def main() -> int:
    if sys.version_info < (3, 11):  # noqa: UP036 - a clear message instead of an obscure failure
        print("slack-agent-controller needs Python 3.11 or newer", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(description="slack-agent-controller: a Claude agent on this machine, driven from Slack")
    ap.add_argument("--env-file", type=Path, default=None, help="default: ~/.config/slack-agent-controller/env")
    ap.add_argument("--check", action="store_true", help="validate config and exit")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    try:
        cfg = load_config(args.env_file)
    except (ConfigError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.state_dir, args.debug)
    log = logging.getLogger("slack-agent-controller")
    log.info("config OK (provider %s, %d allowed users, roots %s, default cwd %s)",
             cfg.provider, len(cfg.allowed_users),
             ",".join(map(str, cfg.cwd_roots)), cfg.default_cwd)
    if args.check:
        print("config OK")
        return 0

    lock = acquire_instance_lock(cfg.state_dir)
    if lock is None:
        return 2  # systemd: RestartPreventExitStatus=2, do not loop

    from slack_agent_controller.app import SlackFrontend

    asyncio.run(SlackFrontend(cfg).run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
