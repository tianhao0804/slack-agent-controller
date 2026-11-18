"""Configuration loading and startup safety checks.

All secrets come from a chmod-600 env file (default ``~/.config/slack-agent-controller/env``).
Nothing in this module ever logs a secret value.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENV_FILE = Path.home() / ".config" / "slack-agent-controller" / "env"

# Env vars that leak in when the bot is started from inside a Claude Code
# session (e.g. during development). They must not reach the child CLI.
_PARENT_SESSION_VARS = {
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_PID",
}
_PARENT_SESSION_PREFIXES = ("CLAUDE_CODE_MESSAGING_",)

PROVIDERS = ("bedrock", "anthropic")
# claude.ai login must not be used by third-party products (Agent SDK terms); never accepted.
_OAUTH_VARS = ("CLAUDE_CODE_OAUTH_TOKEN",)
# Variables that would send requests to a provider other than the configured one.
_OTHER_PROVIDER_VARS = {
    "bedrock": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"),
    "anthropic": ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"),
}

# Paths that are never readable/writable by Claude through this bot.
_DEFAULT_SENSITIVE = [
    "~/.ssh",
    "~/.aws",
    "~/.claude",
    "~/.claude.json",
    "~/.config/slack-agent-controller",
    "~/.local/state/slack-agent-controller",
    "~/.gnupg",
    "~/.netrc",
    "~/.git-credentials",
    "~/.docker/config.json",
    "~/.kube",
    "~/.bash_history",
    "~/.python_history",
]
_DEFAULT_SENSITIVE_GLOBS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".git-credentials",
    ".pgpass",
]


EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODEL_ALIASES = ("opus", "sonnet", "haiku")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unsafe."""


def _resolve(p: str) -> Path:
    return Path(os.path.realpath(os.path.expanduser(p.strip())))


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (file values win).

    Refuses to read the file unless it is owned by us and not group/other
    accessible, so a mis-permissioned token file fails loudly.
    """
    if not path.exists():
        raise ConfigError(f"env file not found: {path} (copy .env.example there, chmod 600)")
    st = path.stat()
    if st.st_uid != os.getuid():
        raise ConfigError(f"env file {path} is not owned by the current user")
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigError(f"env file {path} must be chmod 600 (currently {oct(st.st_mode & 0o777)})")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif value.startswith("#"):
            # "KEY=   # note" -> empty value (the comment is not the value).
            value = ""
        else:
            # Strip inline comments ("VALUE   # note") from unquoted values.
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        os.environ[key] = value


def _csv(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def _float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return float(v)


def _int(name: str, default: int | None) -> int | None:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return int(v)


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    slack_bot_token: str = field(repr=False)
    slack_app_token: str = field(repr=False)
    allowed_users: frozenset[str]
    allowed_channels: frozenset[str]
    cwd_roots: tuple[Path, ...]
    default_cwd: Path
    sensitive_paths: tuple[Path, ...]
    sensitive_globs: tuple[str, ...]
    auto_allow_tools: frozenset[str]
    disallowed_tools: tuple[str, ...]
    approval_timeout_s: float
    state_dir: Path
    env_file: Path
    model: str | None
    max_budget_usd: float | None
    max_turns: int | None
    notify_after_s: float
    idle_disconnect_s: float
    update_interval_s: float
    cli_path: str | None
    scrub_subprocess_env: bool
    load_claude_md: bool
    squeue_path: str | None
    upload_max_mb: float
    token_expires: dt.date | None
    provider: str = "bedrock"               # bedrock | anthropic (more backends may come later)
    node_name: str = "this node"            # shown in messages ("open in a terminal on <node>")
    # Extra read-only locations (e.g. the CLI's own background-task output dir).
    read_only_roots: tuple[Path, ...] = ()
    default_mode: str = "ask"  # "ask" (Slack approval) or "auto" (CLI auto-mode classifier)
    default_effort: str | None = None  # None = Claude Code's default for the model
    learn_approved_commands: bool = True  # offer to remember eligible Bash commands in their working directory
    max_running_turns: int = 4            # replies running at once across all threads (others wait)
    inbox_dir: Path = Path.home() / ".cache" / "slack-agent-controller" / "inbox"  # files you attach in Slack
    jobs_dir: Path = Path.home() / ".local" / "share" / "slack-agent-controller" / "jobs"  # sac-job records and logs
    jobs_poll_s: float = 15.0
    jobs_enabled: bool = False                # sac-job needs systemd user services (Linux)
    inbox_max_mb: float = 20.0
    inbox_keep_days: int = 7
    stall_notify_s: float = 900.0        # @mention when a running reply shows no activity this long
    projects: tuple[tuple[str, str], ...] = ()  # (alias, directory as written) for `!new <alias>` / `!projects`
    max_queued_per_thread: int = 5        # messages waiting behind a running reply in one thread
    # Terminal mirroring (Claude Code hooks -> local socket -> Slack thread).
    mirror_dirs: tuple[Path, ...] = ()  # empty = mirroring disabled
    mirror_channel: str | None = None
    mirror_user: str | None = None      # @mentioned when a terminal waits for permission
    mirror_socket: Path | None = None

    def mirrored(self, p: Path) -> bool:
        return any(p == r or p.is_relative_to(r) for r in self.mirror_dirs)

    def under_roots(self, p: Path) -> bool:
        return any(p == r or p.is_relative_to(r) for r in self.cwd_roots)

    def readable(self, p: Path) -> bool:
        return self.under_roots(p) or any(p == r or p.is_relative_to(r) for r in self.read_only_roots)


def _projects(roots: tuple[Path, ...]) -> tuple[tuple[str, str], ...]:
    """PROJECTS=alias=dir,alias=dir. Directories keep the spelling given (symlinks matter for
    Claude Code transcripts) but must resolve under CWD_ROOTS."""
    out: list[tuple[str, str]] = []
    for item in _csv("PROJECTS"):
        alias, sep, raw = item.partition("=")
        alias, raw = alias.strip().lower(), raw.strip()
        if not sep or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,23}", alias) or not raw:
            raise ConfigError(f"PROJECTS entry {item!r}: use alias=directory (alias: a-z, 0-9, _ or -)")
        given = os.path.abspath(os.path.expanduser(raw))
        real = _resolve(given)
        if not real.is_dir() or not any(real == r or real.is_relative_to(r) for r in roots):
            raise ConfigError(f"PROJECTS {alias}: {real} is not a directory under CWD_ROOTS")
        if alias in (a for a, _ in out):
            raise ConfigError(f"PROJECTS: alias {alias} is listed twice")
        out.append((alias, given))
    return tuple(out)


def default_mirror_socket() -> str:
    """Node-local, user-private socket path shared with deploy/sac_hook.py."""
    run = Path(f"/run/user/{os.getuid()}")
    return str(run / "sac-mirror.sock") if run.is_dir() else "~/.local/state/slack-agent-controller/mirror.sock"


def check_provider() -> str:
    """Validate PROVIDER (bedrock | anthropic) and its credentials; make sure no
    other provider's variables could redirect requests. Returns the provider."""
    raw = os.environ.get("PROVIDER", "").strip().lower()
    if not raw:
        if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
            raw = "bedrock"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            raw = "anthropic"
        else:
            raise ConfigError("Set PROVIDER=bedrock (with AWS credentials) or PROVIDER=anthropic (with ANTHROPIC_API_KEY)")
    if raw not in PROVIDERS:
        raise ConfigError(f"PROVIDER must be one of {', '.join(PROVIDERS)} (got {raw!r})")
    for var in _OAUTH_VARS:
        if os.environ.get(var):
            raise ConfigError(f"{var} is set; claude.ai login is not supported. Use an API key or Amazon Bedrock.")
    if raw == "bedrock":
        os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
    for var in _OTHER_PROVIDER_VARS[raw]:
        if os.environ.get(var):
            raise ConfigError(f"{var} is set but PROVIDER={raw}; refusing to start so requests cannot go elsewhere")
    if raw == "bedrock":
        if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
            raise ConfigError("PROVIDER=bedrock needs AWS_REGION")
        if not (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("AWS_PROFILE")
                or os.environ.get("AWS_ACCESS_KEY_ID")):
            raise ConfigError("PROVIDER=bedrock needs AWS_BEARER_TOKEN_BEDROCK (Bedrock API key), AWS_PROFILE "
                              "or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
            raise ConfigError("PROVIDER=anthropic needs ANTHROPIC_API_KEY (an sk-ant-... key)")
    return raw


def default_node_name() -> str:
    import socket

    return socket.gethostname().split(".")[0] or "this node"


def strip_parent_session_env() -> None:
    for k in list(os.environ):
        if k in _PARENT_SESSION_VARS or k.startswith(_PARENT_SESSION_PREFIXES):
            os.environ.pop(k, None)


def load_config(env_file: Path | None = None) -> Config:
    """Load env file, validate, and return an immutable Config.

    Side effects: populates os.environ from the env file, then removes the
    Slack tokens from os.environ so they are never inherited by the Claude
    CLI or anything it spawns.
    """
    env_path = env_file or Path(os.environ.get("SAC_ENV_FILE", str(DEFAULT_ENV_FILE)))
    env_path = env_path.expanduser()
    load_env_file(env_path)
    strip_parent_session_env()

    provider = check_provider()

    bot_token = os.environ.pop("SLACK_BOT_TOKEN", "")
    app_token = os.environ.pop("SLACK_APP_TOKEN", "")
    if not bot_token.startswith("xoxb-"):
        raise ConfigError("SLACK_BOT_TOKEN missing or not an xoxb- token")
    if not app_token.startswith("xapp-"):
        raise ConfigError("SLACK_APP_TOKEN missing or not an xapp- token")

    allowed_users = frozenset(_csv("ALLOWED_SLACK_USER_IDS"))
    if not allowed_users:
        raise ConfigError("ALLOWED_SLACK_USER_IDS is empty")

    roots = tuple(_resolve(p) for p in _csv("CWD_ROOTS", str(Path.home())))
    for r in roots:
        if not r.is_dir():
            raise ConfigError(f"CWD_ROOTS entry is not a directory: {r}")
    default_cwd = _resolve(os.environ.get("DEFAULT_CWD", "") or str(roots[0]))

    state_dir = _resolve(os.environ.get("STATE_DIR", "") or "~/.local/state/slack-agent-controller")
    sensitive = [_resolve(p) for p in _DEFAULT_SENSITIVE + _csv("EXTRA_SENSITIVE_PATHS")]
    sensitive += [_resolve(str(env_path)), state_dir]
    globs = tuple(_DEFAULT_SENSITIVE_GLOBS + _csv("EXTRA_SENSITIVE_GLOBS"))

    expires_raw = (os.environ.get("API_KEY_EXPIRES") or os.environ.get("BEDROCK_TOKEN_EXPIRES") or "").strip()
    expires = dt.date.fromisoformat(expires_raw) if expires_raw else None

    squeue = os.environ.get("SQUEUE_PATH", "").strip() or shutil.which("squeue")

    inbox = _resolve(os.environ.get("INBOX_DIR", "") or "~/.cache/slack-agent-controller/inbox")
    jobs = _resolve(os.environ.get("JOBS_DIR", "") or "~/.local/share/slack-agent-controller/jobs")
    if any(jobs == s or jobs.is_relative_to(s) for s in sensitive):
        raise ConfigError(f"JOBS_DIR {jobs} is inside a protected path")
    if any(inbox == s or inbox.is_relative_to(s) for s in sensitive):
        raise ConfigError(f"INBOX_DIR {inbox} is inside a protected path; Claude could not read the files")
    if _bool("AUTO_ALLOW_READONLY_BASH", False):
        raise ConfigError("AUTO_ALLOW_READONLY_BASH was removed: the program-name rule could be bypassed with "
                          "option/wildcard variants (see README). Delete it from the env file; in auto mode the "
                          "Claude Code classifier decides, in ask mode you can choose which commands to remember "
                          "(!allowlist).")
    cfg = Config(
        slack_bot_token=bot_token,
        slack_app_token=app_token,
        allowed_users=allowed_users,
        allowed_channels=frozenset(_csv("ALLOWED_CHANNEL_IDS")),
        cwd_roots=roots,
        default_cwd=default_cwd,
        sensitive_paths=tuple(dict.fromkeys(sensitive)),
        sensitive_globs=globs,
        auto_allow_tools=frozenset(_csv("AUTO_ALLOW_EXTRA_TOOLS")),
        disallowed_tools=tuple(_csv("DISALLOWED_TOOLS", "")),
        approval_timeout_s=_float("APPROVAL_TIMEOUT_SEC", 1800.0) or 1800.0,
        state_dir=state_dir,
        env_file=env_path,
        model=os.environ.get("CLAUDE_MODEL", "").strip() or None,
        max_budget_usd=_float("MAX_BUDGET_USD_PER_TURN", 25.0),
        max_turns=_int("MAX_TURNS_PER_PROMPT", None),
        notify_after_s=_float("NOTIFY_AFTER_SEC", 60.0) or 60.0,
        idle_disconnect_s=_float("IDLE_DISCONNECT_SEC", 1800.0) or 1800.0,
        update_interval_s=max(1.0, _float("UPDATE_INTERVAL_SEC", 1.5) or 1.5),
        cli_path=os.environ.get("CLAUDE_CLI_PATH", "").strip() or None,
        scrub_subprocess_env=_bool("SCRUB_SUBPROCESS_ENV", True),
        load_claude_md=_bool("LOAD_CLAUDE_MD", True),
        squeue_path=squeue,
        upload_max_mb=_float("UPLOAD_MAX_MB", 50.0) or 50.0,
        token_expires=expires,
        provider=provider,
        node_name=os.environ.get("NODE_NAME", "").strip() or default_node_name(),
        read_only_roots=(_resolve(f"/tmp/claude-{os.getuid()}"), inbox, jobs),
        jobs_dir=jobs,
        jobs_poll_s=max(2.0, _float("JOBS_POLL_SEC", 15.0) or 15.0),
        jobs_enabled=_bool("JOBS_ENABLED", sys.platform == "linux" and bool(shutil.which("systemd-run"))),
        inbox_dir=inbox,
        inbox_max_mb=_float("INBOX_MAX_MB", 20.0) or 20.0,
        inbox_keep_days=max(1, _int("INBOX_KEEP_DAYS", 7) or 7),
        projects=_projects(roots),
        stall_notify_s=max(60.0, (_float("STALL_NOTIFY_MIN", 15.0) or 15.0) * 60),
        default_mode=os.environ.get("DEFAULT_PERMISSION_MODE", "").strip() or "ask",
        default_effort=os.environ.get("DEFAULT_EFFORT", "").strip().lower() or None,
        learn_approved_commands=_bool("LEARN_APPROVED_COMMANDS", True),
        max_running_turns=max(1, _int("MAX_RUNNING_TURNS", 4) or 4),
        max_queued_per_thread=max(1, _int("MAX_QUEUED_PER_THREAD", 5) or 5),
        mirror_dirs=tuple(_resolve(p) for p in _csv("MIRROR_DIRS")),
        mirror_channel=os.environ.get("MIRROR_CHANNEL", "").strip() or None,
        mirror_user=os.environ.get("MIRROR_USER", "").strip() or None,
        mirror_socket=_resolve(os.environ.get("MIRROR_SOCKET", "") or default_mirror_socket()),
    )
    if cfg.mirror_dirs and sys.platform != "linux":
        raise ConfigError("terminal mirroring (MIRROR_DIRS) needs Linux (Unix-socket peer credentials, /proc)")
    if cfg.jobs_enabled and not shutil.which("systemd-run"):
        raise ConfigError("JOBS_ENABLED=1 needs systemd user services (systemd-run); set JOBS_ENABLED=0")
    if cfg.mirror_dirs:
        for d in cfg.mirror_dirs:
            if not d.is_dir() or not cfg.under_roots(d):
                raise ConfigError(f"MIRROR_DIRS entry {d} is not a directory under CWD_ROOTS")
        if not cfg.mirror_channel:
            raise ConfigError("MIRROR_DIRS is set but MIRROR_CHANNEL is empty")
        if cfg.mirror_channel not in cfg.allowed_channels:
            raise ConfigError("MIRROR_CHANNEL must also be listed in ALLOWED_CHANNEL_IDS")
        if not cfg.mirror_user or cfg.mirror_user not in cfg.allowed_users:
            raise ConfigError("MIRROR_USER must be one of ALLOWED_SLACK_USER_IDS")
    if cfg.default_effort and cfg.default_effort not in EFFORTS:
        raise ConfigError(f"DEFAULT_EFFORT must be one of {', '.join(EFFORTS)}")
    if cfg.default_mode not in ("ask", "auto"):
        raise ConfigError("DEFAULT_PERMISSION_MODE must be 'ask' or 'auto'")
    # Tools with side effects must always go through Slack approval.
    unsafe = {"Bash", "Edit", "Write", "MultiEdit", "NotebookEdit", "WebFetch", "KillShell", "TaskStop"}
    bad = sorted(t for t in cfg.auto_allow_tools if t in unsafe or not re.fullmatch(r"[A-Za-z_][\w-]*", t))
    if bad:
        raise ConfigError(f"AUTO_ALLOW_EXTRA_TOOLS may only list read-only tools; rejected: {bad}")
    if cfg.cli_path and not os.access(cfg.cli_path, os.X_OK):
        raise ConfigError(f"CLAUDE_CLI_PATH is not an executable file: {cfg.cli_path}")
    if os.environ.get("SQUEUE_PATH", "").strip() and not (cfg.squeue_path and os.access(cfg.squeue_path, os.X_OK)):
        raise ConfigError(f"SQUEUE_PATH is not an executable file: {cfg.squeue_path}")
    if not cfg.under_roots(default_cwd) or not default_cwd.is_dir():
        raise ConfigError(f"DEFAULT_CWD {default_cwd} is not a directory under CWD_ROOTS")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
    return cfg
