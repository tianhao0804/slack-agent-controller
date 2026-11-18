"""Integration: Claude submits a sac-job from a Slack session; real systemd units; the watcher reports
in the thread; Ask-Claude and Cancel buttons (real Bedrock/Haiku, fake Slack, fake nvidia-smi).

Run: .venv/bin/python -m tests.integration_jobs
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from slack_agent_controller.config import strip_parent_session_env
from tests.helpers import FakeWeb, is_reply, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"


async def wait_for(cond, timeout: float = 120) -> bool:  # noqa: ANN001
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.5)
    return False


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_jobs_"))
    work, jobs = tmp / "proj", tmp / "jobs"
    work.mkdir()
    smi = tmp / "smi"
    smi.write_text("#!/bin/sh\necho '0, 97, 40000'\necho '1, 0, 3'\n")
    smi.chmod(0o755)
    os.environ.update(SAC_JOBS_DIR=str(jobs), SAC_JOB_NVIDIA_SMI=str(smi), SAC_JOB_POLL_S="1")
    from slack_agent_controller.app import SlackFrontend
    from slack_agent_controller.jobs import JobWatcher

    cfg = make_config(tmp, roots=[tmp], model="haiku", default_cwd=work.resolve(), jobs_dir=jobs, jobs_enabled=True,
                      read_only_roots=(jobs,), jobs_poll_s=1.0)
    f = SlackFrontend(cfg)
    web = FakeWeb()
    f.hub.web = web  # type: ignore[assignment]

    async def approve(sess, tool, tin, ctx, reason):  # noqa: ANN001
        return "approve", "UOWNER"

    f.hub.request_approval = approve  # type: ignore[method-assign]
    f.jobs = JobWatcher(f)
    f.jobs.cli.NVIDIA_SMI = str(smi)
    watcher = asyncio.create_task(f.jobs.loop())
    s = f.hub.create_session("D1", "1.0", work.resolve(), "UOWNER")
    code = "import os,time; print('GPU', os.environ['CUDA_VISIBLE_DEVICES']); time.sleep(4); print('RESULT 42')"
    s.submit(f'Start this as a tracked job named demo using 1 GPU: python3 -c "{code}". Then tell me the job id '
             "and stop; do not wait for it.", "UOWNER")
    await wait_idle(s, 200)
    ids = [p.name for p in jobs.iterdir() if p.is_dir() and len(p.name) == 6] if jobs.exists() else []
    check("Claude submitted one job", len(ids) == 1, str(ids))
    if not ids:
        return 1
    jid = ids[0]
    meta = json.loads((jobs / jid / "meta.json").read_text())
    env = json.loads((jobs / jid / "env.json").read_text())
    check("job tied to the Slack session", meta.get("session_id") == s.rec.session_id)
    check("job asked for 1 GPU", meta.get("gpus") == 1, str(meta.get("gpus")))
    check("no credentials in the job environment",
          not any(k.startswith(("AWS_BEARER", "AWS_SECRET", "SLACK_", "ANTHROPIC_")) for k in env))
    done = await wait_for(lambda: f.jobs.state.get(jid, {}).get("done"), 90)
    check("status message in the thread", any(f"`{jid}`" in kw.get("text", "") and kw.get("thread_ts") == "1.0"
                                              for op, kw in web.calls if op == "post"))
    finish = [kw for op, kw in web.calls if op == "post" and "sac_jobdone_" + jid in json.dumps(kw.get("blocks", []))]
    body = json.dumps(finish[-1]["blocks"]) if finish else ""
    check("finish message with the result", done and "RESULT 42" in body and "GPU 1" in body, body[:300])

    # Ask Claude to look at it
    btn = {"action_id": "sac_job_analyze", "value": jid}
    await f.jobs.on_action({"user": {"id": "UOWNER"}, "channel": {"id": "D1"}, "message": {"ts": "9"}}, btn)
    await wait_idle(s, 200)
    last = [kw["text"] for op, kw in web.calls if op == "update" and is_reply(kw)][-1]
    audit = [json.loads(x) for x in (cfg.state_dir / "audit.jsonl").read_text().splitlines() if x.strip()]
    log_path = str(jobs / jid / "log")
    read_it = any(a.get("tool") in ("Read", "Bash") and (log_path in str(a.get("input")) or f"log {jid}" in str(a.get("input")))
                  for a in audit)
    # The wording of Claude's summary varies; that it opened the job's log is the reliable signal.
    check("Claude opened the job log", read_it, last[-200:])
    print(f"  (Claude's summary mentions the result value: {'42' in last})")

    # Cancel button on a long job
    os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
    import subprocess
    r = subprocess.run([str(Path(__file__).resolve().parent.parent / "deploy" / "sac-job"), "submit", "--name", "long",
                        "--cwd", str(work), "--", "sleep", "300"], capture_output=True, text=True, env=dict(os.environ))
    long_id = r.stdout.split()[1] if r.returncode == 0 else ""
    running = await wait_for(lambda: f.jobs.cli.read_status(long_id).get("state") == "running"
                             and long_id in f.jobs.state, 30)
    check("terminal-style job tracked", running, r.stdout + r.stderr)
    await f.jobs.on_action({"user": {"id": "UOWNER"}, "channel": {"id": "D1"}, "message": {"ts": "9"}},
                           {"action_id": "sac_job_cancel", "value": long_id})
    ok = await wait_for(lambda: f.jobs.state.get(long_id, {}).get("done"), 30)
    check("Cancel button stopped the job", ok and f.jobs.cli.read_status(long_id).get("state") == "cancelled")

    watcher.cancel()
    await f.hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
