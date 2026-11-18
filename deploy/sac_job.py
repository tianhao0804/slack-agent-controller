#!/usr/bin/env python3
"""sac-job: run long computations as their own systemd user units, tracked by the Slack bot.

    sac-job submit [--name N] [--gpus K] [--cwd DIR] [--progress-min M] -- COMMAND [ARGS...]
    sac-job list [--all]
    sac-job status JOB
    sac-job log JOB [-n LINES]
    sac-job cancel JOB

A job runs in a transient unit ``sac-job-<id>`` (``systemd-run --user``), so it
keeps running when Claude, the bot or your terminal stops; it ends only when
it finishes, is cancelled, or the node reboots. With ``--gpus K`` it waits in
a first-come-first-served queue until K GPUs are free (utilization below
SAC_JOB_GPU_UTIL_MAX %, memory below SAC_JOB_GPU_MEM_MAX_MB, and not given to
another job) and runs with CUDA_VISIBLE_DEVICES set to them. ``--gpus 0`` (the
default) is a CPU job and gets no GPU.

The job inherits the caller's environment minus anything that looks like a
credential (tokens, keys, passwords, AWS_*/ANTHROPIC_*/CLAUDE_*/SLACK_*).
Everything lives in ``~/.local/share/slack-agent-controller/jobs/<id>/`` (meta.json,
status.json, log). The Slack bot reports progress and the result in the
Slack thread of the Claude session that submitted it (CLAUDE_CODE_SESSION_ID);
jobs submitted from a plain terminal are reported in the bot's channel.

Standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

JOBS_DIR = Path(os.environ.get("SAC_JOBS_DIR") or Path.home() / ".local" / "share" / "slack-agent-controller" / "jobs")
NVIDIA_SMI = os.environ.get("SAC_JOB_NVIDIA_SMI") or "nvidia-smi"
POLL_S = float(os.environ.get("SAC_JOB_POLL_S") or 30)
GPU_UTIL_MAX = float(os.environ.get("SAC_JOB_GPU_UTIL_MAX") or 10)
GPU_MEM_MAX_MB = float(os.environ.get("SAC_JOB_GPU_MEM_MAX_MB") or 1024)
ACTIVE = ("queued", "running")
_SECRET_ENV = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_?KEY|PRIVATE|SESSION_KEY"
                         r"|^(AWS_|ANTHROPIC_|CLAUDE|SLACK_|SAC_|DBUS_SESSION|SSH_AUTH_SOCK|GPG_AGENT)", re.I)
_ID = re.compile(r"^[0-9a-f]{6}$")


# -- files -------------------------------------------------------------------
def job_dir(jid: str) -> Path:
    if not _ID.match(jid):
        raise SystemExit(f"sac-job: {jid!r} is not a job id (6 hex characters; see `sac-job list`)")
    return JOBS_DIR / jid


def _write(path: Path, data: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _read(path: Path, default: object = None) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def read_status(jid: str) -> dict:
    s = _read(job_dir(jid) / "status.json", {})
    return s if isinstance(s, dict) else {}


def update_status(jid: str, **fields: object) -> dict:
    s = read_status(jid)
    s.update(fields)
    _write(job_dir(jid) / "status.json", s)
    return s


def read_meta(jid: str) -> dict:
    m = _read(job_dir(jid) / "meta.json", {})
    return m if isinstance(m, dict) else {}


def all_jobs() -> list[str]:
    if not JOBS_DIR.is_dir():
        return []
    return sorted((p.name for p in JOBS_DIR.iterdir() if p.is_dir() and _ID.match(p.name)),
                  key=lambda j: float(read_meta(j).get("submitted") or 0))


@contextlib.contextmanager
def gpu_lock():
    JOBS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(JOBS_DIR / "gpu.lock", "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def config() -> dict:
    """Written by the bot at startup: allowed root directories."""
    c = _read(JOBS_DIR / "config.json", {})
    return c if isinstance(c, dict) else {}


# -- GPUs --------------------------------------------------------------------
def gpu_state() -> list[tuple[int, float, float]]:
    """[(index, utilization %, memory used MB)]; empty if nvidia-smi is unavailable."""
    try:
        out = subprocess.run([NVIDIA_SMI, "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            rows.append((int(f[0]), float(f[1]), float(f[2])))
        except (ValueError, IndexError):
            continue
    return rows


def is_live(jid: str) -> bool:
    """The job's runner process still exists (same process: its command line names the job)."""
    pid = read_status(jid).get("runner_pid")
    if not isinstance(pid, int):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"_run" in cmdline and jid.encode() in cmdline


def reap_lost() -> list[str]:
    """Mark queued/running jobs whose runner is gone (node reboot, kill -9) as lost."""
    lost = []
    for j in all_jobs():
        st = read_status(j)
        if st.get("state") in ACTIVE and st.get("runner_pid") and not is_live(j):
            update_status(j, state="lost", ended=time.time(), reason="the job's process is gone (node reboot?)")
            lost.append(j)
    return lost


def _claims() -> dict[str, str]:
    """GPU index -> job id, dropping claims of jobs that are no longer running."""
    c = _read(JOBS_DIR / "gpu_claims.json", {})
    claims = {str(k): str(v) for k, v in (c.items() if isinstance(c, dict) else [])}
    return {g: j for g, j in claims.items() if _ID.match(j) and read_status(j).get("state") == "running"}


def try_claim(jid: str, need: int) -> tuple[list[int] | None, str]:
    """Under the lock: give ``need`` free GPUs to ``jid`` if it is first in line."""
    with gpu_lock():
        reap_lost()
        claims = _claims()
        waiting = [j for j in all_jobs() if read_status(j).get("state") == "queued" and int(read_meta(j).get("gpus") or 0) > 0]
        if waiting and waiting[0] != jid:
            ahead = waiting.index(jid) if jid in waiting else len(waiting)
            return None, f"{ahead} job(s) ahead in the GPU queue"
        gpus = gpu_state()
        if not gpus:
            return None, "nvidia-smi gave no GPU information"
        free = [i for i, util, mem in gpus if util < GPU_UTIL_MAX and mem < GPU_MEM_MAX_MB and str(i) not in claims]
        if len(free) < need:
            return None, f"waiting for {need} free GPU(s); {len(free)} free now"
        picked = free[:need]
        claims.update({str(i): jid for i in picked})
        _write(JOBS_DIR / "gpu_claims.json", claims)
        update_status(jid, state="running", gpus=picked, started=time.time(), reason=None)
        return picked, ""


def release(jid: str) -> None:
    with gpu_lock():
        c = _read(JOBS_DIR / "gpu_claims.json", {})
        if isinstance(c, dict):
            _write(JOBS_DIR / "gpu_claims.json", {k: v for k, v in c.items() if v != jid})


# -- submit ------------------------------------------------------------------
def clean_env(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if not _SECRET_ENV.search(k)}


def under(p: Path, roots: list[str]) -> bool:
    rp = os.path.realpath(p)
    return any(rp == os.path.realpath(r) or rp.startswith(os.path.realpath(r).rstrip("/") + "/") for r in roots)


def submit(a: argparse.Namespace) -> int:
    cmd = list(a.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("sac-job submit: give the command after `--`, e.g. sac-job submit --gpus 1 -- python train.py",
              file=sys.stderr)
        return 2
    cfg = config()
    roots = [str(r) for r in cfg.get("roots") or []]
    cwd = Path(a.cwd or os.getcwd()).expanduser()
    if not cwd.is_dir():
        print(f"sac-job submit: {cwd} is not a directory", file=sys.stderr)
        return 2
    if not roots:
        print("sac-job submit: the Slack bot has not written its job config yet (is it running?)", file=sys.stderr)
        return 2
    if not under(cwd, roots):
        print(f"sac-job submit: {cwd} is outside the allowed directories ({', '.join(roots)})", file=sys.stderr)
        return 2
    total = len(gpu_state())
    if a.gpus > 0 and total == 0:
        print("sac-job submit: no GPUs found (nvidia-smi gave no output); use --gpus 0 for a CPU job", file=sys.stderr)
        return 2
    if a.gpus < 0 or a.gpus > total:
        print(f"sac-job submit: --gpus must be between 0 and {total}", file=sys.stderr)
        return 2
    jid = secrets.token_hex(3)
    while (JOBS_DIR / jid).exists():
        jid = secrets.token_hex(3)
    d = JOBS_DIR / jid
    d.mkdir(parents=True, mode=0o700)
    os.chmod(JOBS_DIR, 0o700)
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID") or None
    meta = {
        "id": jid, "name": (a.name or Path(cmd[0]).name)[:60], "command": cmd, "cwd": str(cwd),
        "gpus": a.gpus, "submitted": time.time(), "session_id": sid,
        "origin": "claude" if sid else "terminal", "progress_min": a.progress_min,
        "user": os.environ.get("USER", ""), "host": os.uname().nodename,
    }
    _write(d / "meta.json", meta)
    _write(d / "env.json", clean_env(dict(os.environ)))
    _write(d / "status.json", {"state": "queued", "reason": "starting"})
    unit = f"sac-job-{jid}"
    run = [os.environ.get("SAC_JOB_SYSTEMD_RUN") or "systemd-run", "--user", "--quiet", "--collect",
           f"--unit={unit}", f"--description=sac-job {meta['name']}"]
    for k in ("SAC_JOBS_DIR", "SAC_JOB_NVIDIA_SMI", "SAC_JOB_POLL_S", "SAC_JOB_GPU_UTIL_MAX", "SAC_JOB_GPU_MEM_MAX_MB"):
        if os.environ.get(k):
            run.append(f"--setenv={k}={os.environ[k]}")
    run += ["--", sys.executable, "-I", str(Path(__file__).resolve()), "_run", jid]
    r = subprocess.run(run, capture_output=True, text=True)
    if r.returncode != 0:
        update_status(jid, state="failed", reason=f"could not start: {r.stderr.strip()[:300]}", ended=time.time())
        print(f"sac-job submit: systemd-run failed: {r.stderr.strip()}", file=sys.stderr)
        return 1
    update_status(jid, unit=unit)
    gp = f"{a.gpus} GPU(s), picked automatically when free" if a.gpus else "CPU only"
    print(f"job {jid} submitted: {meta['name']} · {gp}\n"
          f"  log: {d / 'log'}\n"
          f"  the Slack bot reports progress and the result{' in this thread' if sid else ''}; "
          f"`sac-job status {jid}` to check, `sac-job cancel {jid}` to stop")
    return 0


# -- runner (inside the systemd unit) ------------------------------------------
def runner(jid: str) -> int:
    meta = read_meta(jid)
    env = _read(job_dir(jid) / "env.json", {})
    env = dict(env) if isinstance(env, dict) else {}
    child: dict[str, subprocess.Popen[bytes]] = {}
    cancelled = {"flag": False}

    def on_term(signum: int, _frame: object) -> None:
        # Only signal here; the main flow (waiting on the child) does the bookkeeping.
        cancelled["flag"] = True
        p = child.get("p")
        if p is None:  # still queued: nothing is running yet
            update_status(jid, state="cancelled", ended=time.time(), reason="cancelled")
            release(jid)
            sys.exit(143)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    update_status(jid, state="queued", runner_pid=os.getpid(), queued_at=time.time())
    need = int(meta.get("gpus") or 0)
    gpus: list[int] = []
    if need:
        while True:
            picked, why = try_claim(jid, need)
            if picked is not None:
                gpus = picked
                break
            update_status(jid, reason=why)
            time.sleep(POLL_S)
    else:
        update_status(jid, state="running", gpus=[], started=time.time(), reason=None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["SAC_JOB_ID"] = jid
    logf = job_dir(jid) / "log"
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab", buffering=0) as log:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"[sac-job {jid}] {stamp} start on {'GPU ' + env['CUDA_VISIBLE_DEVICES'] if gpus else 'CPU'}: "
                  f"{' '.join(meta.get('command') or [])}\n".encode())
        try:
            p = subprocess.Popen(meta["command"], cwd=meta["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            log.write(f"[sac-job {jid}] could not start: {e}\n".encode())
            update_status(jid, state="failed", exit_code=127, ended=time.time(), reason=str(e)[:300])
            release(jid)
            return 127
        child["p"] = p
        update_status(jid, pid=p.pid)
        code = p.wait()
        if cancelled["flag"]:
            log.write(f"[sac-job {jid}] {dt.datetime.now():%Y-%m-%d %H:%M:%S} cancelled\n".encode())
        else:
            log.write(f"[sac-job {jid}] {dt.datetime.now():%Y-%m-%d %H:%M:%S} exit {code}\n".encode())
    if cancelled["flag"]:
        update_status(jid, state="cancelled", exit_code=code, ended=time.time(), reason="cancelled")
    else:
        update_status(jid, state="finished" if code == 0 else "failed", exit_code=code, ended=time.time())
    release(jid)
    return 143 if cancelled["flag"] else 0


# -- inspection ----------------------------------------------------------------
def _dur(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def describe(jid: str) -> str:
    m, s = read_meta(jid), read_status(jid)
    state = s.get("state", "?")
    start = s.get("started")
    end = s.get("ended") or time.time()
    took = f" · {_dur(end - start)}" if start else ""
    gp = f" · GPU {','.join(map(str, s['gpus']))}" if s.get("gpus") else (f" · wants {m.get('gpus')} GPU" if m.get("gpus") else "")
    extra = f" · exit {s['exit_code']}" if "exit_code" in s else ""
    why = f" ({s['reason']})" if s.get("reason") and state == "queued" else ""
    return f"{jid}  {state:<9} {m.get('name', '')}{gp}{took}{extra}{why}"


def unit_active(jid: str) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", f"sac-job-{jid}"], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating", "reloading")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sac-job", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="start a job (waits for free GPUs if --gpus > 0)")
    s.add_argument("--name")
    s.add_argument("--gpus", type=int, default=0, help="number of GPUs, picked automatically (default 0 = CPU)")
    s.add_argument("--cwd", help="working directory (default: current)")
    s.add_argument("--progress-min", type=int, default=30, help="progress update interval in Slack (minutes)")
    s.add_argument("command", nargs=argparse.REMAINDER)
    ls = sub.add_parser("list", help="recent jobs")
    ls.add_argument("--all", action="store_true")
    for name in ("status", "cancel"):
        sub.add_parser(name).add_argument("job")
    lg = sub.add_parser("log", help="last lines of a job's output")
    lg.add_argument("job")
    lg.add_argument("-n", type=int, default=40)
    sub.add_parser("_run").add_argument("job")
    a = ap.parse_args(argv)

    if a.cmd == "submit":
        return submit(a)
    if a.cmd == "_run":
        return runner(a.job)
    if a.cmd == "list":
        jobs = all_jobs()
        if not a.all:
            jobs = [j for j in jobs if read_status(j).get("state") in ACTIVE] + \
                   [j for j in jobs if read_status(j).get("state") not in ACTIVE][-10:]
        print("\n".join(describe(j) for j in jobs) or "no jobs")
        return 0
    jid = a.job
    if not job_dir(jid).is_dir():
        print(f"sac-job: no job {jid}", file=sys.stderr)
        return 1
    if a.cmd == "status":
        print(describe(jid))
        print(json.dumps({"meta": read_meta(jid), "status": read_status(jid)}, indent=1))
        return 0
    if a.cmd == "log":
        f = job_dir(jid) / "log"
        lines = f.read_text(errors="replace").splitlines()[-max(1, a.n):] if f.exists() else ["(no output yet)"]
        print("\n".join(lines))
        return 0
    if a.cmd == "cancel":
        state = read_status(jid).get("state")
        if state not in ACTIVE:
            print(f"job {jid} is already {state}")
            return 0
        if unit_active(jid):
            subprocess.run(["systemctl", "--user", "stop", f"sac-job-{jid}"], check=False)
        if read_status(jid).get("state") in ACTIVE:  # the unit was already gone
            update_status(jid, state="cancelled", ended=time.time(), reason="cancelled")
            release(jid)
        print(f"job {jid} cancelled")
        return 0
    return 2


if __name__ == "__main__":
    if shutil.which("systemctl") is None and not os.environ.get("SAC_JOB_SYSTEMD_RUN"):
        print("sac-job needs systemd user services (systemctl)", file=sys.stderr)
        sys.exit(1)
    sys.exit(main())
