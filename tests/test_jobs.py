"""sac-job (CLI + runner logic) and the bot's job watcher (offline; no systemd units started)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from slack_agent_controller.allowlist import JOB_READONLY
from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.hub import _job_note
from slack_agent_controller.jobs import JobWatcher, load_cli
from tests.helpers import FakeWeb, make_config


@pytest.fixture
def cli(tmp_path: Path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    smi = tmp_path / "smi"
    smi.write_text("#!/bin/sh\necho '0, 95, 30000'\necho '1, 0, 5'\necho '2, 3, 100'\n")
    smi.chmod(0o755)
    monkeypatch.setenv("SAC_JOB_NVIDIA_SMI", str(smi))
    mod = load_cli(jobs)
    monkeypatch.setattr(mod, "NVIDIA_SMI", str(smi))
    (jobs / "config.json").write_text(json.dumps({"roots": [str(tmp_path)]}))
    return mod


def make_job(cli, jid: str, gpus: int = 0, state: str = "queued", submitted: float | None = None,
             session: str | None = None, **status: Any) -> Path:
    d = cli.JOBS_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"id": jid, "name": f"job-{jid}", "command": ["python", "run.py"],
                                             "cwd": str(cli.JOBS_DIR.parent), "gpus": gpus,
                                             "submitted": submitted or time.time(), "session_id": session,
                                             "origin": "claude" if session else "terminal", "progress_min": 1}))
    (d / "status.json").write_text(json.dumps({"state": state, **status}))
    return d


# -- CLI --------------------------------------------------------------------------
def test_clean_env_drops_credentials(cli):
    env = {"PATH": "/bin", "HOME": "/h", "VIRTUAL_ENV": "/v", "AWS_BEARER_TOKEN_BEDROCK": "x", "AWS_REGION": "r",
           "MY_API_KEY": "x", "GITHUB_TOKEN": "x", "DB_PASSWORD": "x", "CLAUDE_CODE_SESSION_ID": "s",
           "SLACK_BOT_TOKEN": "x", "SSH_AUTH_SOCK": "/s", "OMP_NUM_THREADS": "8"}
    assert cli.clean_env(env) == {"PATH": "/bin", "HOME": "/h", "VIRTUAL_ENV": "/v", "OMP_NUM_THREADS": "8"}


def test_gpu_picking_is_fifo_and_respects_claims(cli):
    me = os.getpid()
    make_job(cli, "aaaaaa", gpus=1, submitted=1.0, runner_pid=me)
    make_job(cli, "bbbbbb", gpus=1, submitted=2.0, runner_pid=me)
    cli.is_live = lambda j: True                      # pretend both runners are alive
    picked, why = cli.try_claim("bbbbbb", 1)
    assert picked is None and "ahead" in why, "the older job goes first"
    picked, _ = cli.try_claim("aaaaaa", 1)
    assert picked == [1], "GPU 0 is busy (95%), GPU 1 is free"
    picked, _ = cli.try_claim("bbbbbb", 1)
    assert picked == [2], "GPU 1 is claimed by job a; GPU 2 (3%, 100 MB) is free"
    make_job(cli, "cccccc", gpus=1, submitted=3.0, runner_pid=me)
    picked, why = cli.try_claim("cccccc", 1)
    assert picked is None and "0 free" in why
    cli.update_status("aaaaaa", state="finished")     # a finished job's claim no longer counts
    picked, _ = cli.try_claim("cccccc", 1)
    assert picked == [1]


def test_lost_jobs_are_reaped_and_do_not_block_the_queue(cli):
    dead = subprocess.Popen(["true"])
    dead.wait()
    make_job(cli, "dddddd", gpus=1, submitted=1.0, runner_pid=dead.pid)        # e.g. before a node reboot
    make_job(cli, "eeeeee", gpus=1, submitted=2.0, runner_pid=os.getpid())
    real_live = cli.is_live
    cli.is_live = lambda j: j == "eeeeee" or real_live(j)
    picked, _ = cli.try_claim("eeeeee", 1)
    assert picked == [1]
    assert cli.read_status("dddddd")["state"] == "lost"


def test_submit_validation(cli, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SAC_JOB_SYSTEMD_RUN", "true")       # do not start a real unit
    work = tmp_path / "w"
    work.mkdir()
    assert cli.main(["submit", "--cwd", str(work), "--", "echo", "hi"]) == 0
    jid = [p.name for p in cli.JOBS_DIR.iterdir() if p.is_dir()][0]
    meta = cli.read_meta(jid)
    assert meta["command"] == ["echo", "hi"] and meta["gpus"] == 0 and meta["cwd"] == str(work)
    assert oct((cli.JOBS_DIR / jid / "env.json").stat().st_mode & 0o777) == "0o600"
    assert cli.main(["submit", "--cwd", "/etc", "--", "true"]) == 2          # outside the roots
    assert cli.main(["submit", "--gpus", "9", "--cwd", str(work), "--", "true"]) == 2   # only 3 GPUs
    assert cli.main(["submit", "--cwd", str(work)]) == 2                    # no command
    with pytest.raises(SystemExit):
        cli.main(["status", "../x"])


def test_runner_runs_cpu_job_and_records_exit(cli, tmp_path: Path):
    jid = "f00001"
    make_job(cli, jid)
    (cli.JOBS_DIR / jid / "meta.json").write_text(json.dumps({
        "id": jid, "name": "t", "command": [sys.executable, "-c",
                                            "import os; print('cuda=[%s]' % os.environ['CUDA_VISIBLE_DEVICES']); raise SystemExit(3)"],
        "cwd": str(tmp_path), "gpus": 0, "submitted": time.time()}))
    (cli.JOBS_DIR / jid / "env.json").write_text(json.dumps({"PATH": os.environ["PATH"]}))
    assert cli.runner(jid) == 0
    st = cli.read_status(jid)
    assert st["state"] == "failed" and st["exit_code"] == 3
    assert "cuda=[]" in (cli.JOBS_DIR / jid / "log").read_text()


# -- watcher ---------------------------------------------------------------------------
def _front(tmp: Path, cli, **over: Any) -> SlackFrontend:
    cfg = make_config(tmp, [tmp], jobs_dir=cli.JOBS_DIR, mirror_channel="CMIR", mirror_user="UOWNER",
                      allowed_channels=frozenset({"CMIR"}), **over)
    f = SlackFrontend(cfg)
    f.hub.web = FakeWeb()  # type: ignore[assignment]
    f.jobs = JobWatcher(f)
    f.jobs.cli = cli
    return f


def _posts(f: SlackFrontend) -> list[dict]:
    return [kw for op, kw in f.hub.web.calls if op == "post"]  # type: ignore[attr-defined]


def test_watcher_reports_a_job_in_its_thread_from_start_to_end(tmp_path: Path, cli):
    async def run():
        f = _front(tmp_path, cli)
        f.jobs._first_run = False
        sess = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        d = make_job(cli, "a1b2c3", gpus=1, session=sess.rec.session_id, reason="waiting for 1 free GPU(s); 0 free now")
        await f.jobs.scan()
        first = _posts(f)[-1]
        assert first["thread_ts"] == "1.0" and "queued" in first["text"] and "waiting for 1 free" in first["text"]
        assert [e["action_id"] for e in first["blocks"][1]["elements"]] == ["sac_job_log", "sac_job_cancel"]

        cli.update_status("a1b2c3", state="running", gpus=[1], started=time.time() - 120)
        (d / "log").write_text("epoch 1 loss 0.9\nepoch 2 loss nan\n")
        await f.jobs.scan()
        upd = [kw for op, kw in f.hub.web.calls if op == "update"][-1]  # type: ignore[attr-defined]
        assert "running" in upd["text"] and "GPU 1" in upd["text"] and "epoch 2" in upd["text"]
        alerts = [p for p in _posts(f) if ":rotating_light:" in p["text"]]
        assert len(alerts) == 1 and "loss nan" in alerts[0]["text"]
        with (d / "log").open("a") as fh:
            fh.write("epoch 3 loss nan\n")
        await f.jobs.scan()
        assert len([p for p in _posts(f) if ":rotating_light:" in p["text"]]) == 1, "one alert per kind"

        cli.update_status("a1b2c3", state="finished", exit_code=0, ended=time.time())
        await f.jobs.scan()
        done = _posts(f)[-1]
        assert "<@UOWNER>" in done["blocks"][0]["text"]["text"] and "finished after" in done["blocks"][0]["text"]["text"]
        assert [e["action_id"] for e in done["blocks"][1]["elements"]] == ["sac_job_full", "sac_job_analyze"]
        n = len(f.hub.web.calls)  # type: ignore[attr-defined]
        await f.jobs.scan()
        assert len(f.hub.web.calls) == n, "nothing more once done"  # type: ignore[attr-defined]

        again = JobWatcher(f)                                # bot restart
        again.cli = cli
        await again.scan()
        assert len(f.hub.web.calls) == n  # type: ignore[attr-defined]
    asyncio.run(run())


def test_progress_updates_follow_the_interval(tmp_path: Path, cli):
    async def run():
        f = _front(tmp_path, cli)
        f.jobs._first_run = False
        make_job(cli, "0000aa", state="running", started=time.time())
        await f.jobs.scan()
        st = f.jobs.state["0000aa"]
        ups = lambda: len([1 for op, _ in f.hub.web.calls if op == "update"])  # type: ignore[attr-defined]  # noqa: E731
        n = ups()
        await f.jobs.scan()
        assert ups() == n, "no update before progress_min"
        st["progress_at"] -= 61
        await f.jobs.scan()
        assert ups() == n + 1
    asyncio.run(run())


def test_terminal_job_gets_its_own_thread_and_old_jobs_are_skipped(tmp_path: Path, cli):
    async def run():
        make_job(cli, "01d001", state="finished", ended=1.0)   # existed before the watcher ran
        f = _front(tmp_path, cli)
        await f.jobs.scan()
        assert not _posts(f)
        make_job(cli, "7e0001", state="running", started=time.time())
        await f.jobs.scan()
        roots = [p for p in _posts(f) if p.get("thread_ts") is None]
        assert len(roots) == 1 and roots[0]["channel"] == "CMIR" and "from a terminal" in roots[0]["text"]
        cli.update_status("7e0001", state="failed", exit_code=1, ended=time.time())
        await f.jobs.scan()
        done = _posts(f)[-1]
        ids = [e["action_id"] for e in done["blocks"][1]["elements"]]
        assert ids == ["sac_job_full"], "no Claude session to ask"
        assert "failed (exit 1)" in done["blocks"][0]["text"]["text"]
    asyncio.run(run())


def test_job_buttons(tmp_path: Path, cli):
    async def run():
        f = _front(tmp_path, cli)
        f.jobs._first_run = False
        sess = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        d = make_job(cli, "b0b0b0", state="failed", session=sess.rec.session_id, exit_code=1,
                     started=time.time() - 5, ended=time.time())
        (d / "log").write_text("Traceback (most recent call last):\nValueError: bad input\n")
        await f.jobs.scan()
        body = {"user": {"id": "UOWNER"}, "channel": {"id": "D1"}, "message": {"ts": "9.0"}}
        await f.jobs.on_action(body, {"action_id": "sac_job_log", "value": "b0b0b0"})
        assert "ValueError: bad input" in _posts(f)[-1]["text"]
        sent: list[str] = []

        async def fake_prompt(channel, ts, thread_ts, user, text, files=None):  # noqa: ANN001
            sent.append(text)

        f._prompt = fake_prompt  # type: ignore[method-assign]
        await f.jobs.on_action(body, {"action_id": "sac_job_analyze", "value": "b0b0b0"})
        assert sent and "b0b0b0" in sent[0] and str(d / "log") in sent[0]
        cancelled: list[str] = []
        f.jobs.cancel = lambda jid, user: cancelled.append(jid) or "job cancelled"  # type: ignore[method-assign]
        await f.jobs.on_action(body, {"action_id": "sac_job_cancel_hb0b0b0", "value": "b0b0b0"})
        assert cancelled == ["b0b0b0"]
        n = len(f.hub.web.calls)  # type: ignore[attr-defined]
        await f.jobs.on_action({**body, "user": {"id": "UEVIL"}}, {"action_id": "sac_job_log", "value": "b0b0b0"})
        await f.jobs.on_action(body, {"action_id": "sac_job_log", "value": "../etc"})
        assert len(f.hub.web.calls) == n  # type: ignore[attr-defined]
    asyncio.run(run())


# -- permissions and display --------------------------------------------------------------
@pytest.mark.parametrize("cmd,ok", [
    ("sac-job list", True), ("sac-job list --all", True), ("/home/u/slack-agent-controller/deploy/sac-job status a1b2c3", True),
    ("sac-job log a1b2c3 -n 100", True), ("sac-job submit -- python x.py", False), ("sac-job cancel a1b2c3", False),
    ("sac-job log a1b2c3; rm -rf x", False), ("sac-job status ../../x", False),
])
def test_read_only_job_commands_need_no_card(cmd, ok):
    assert bool(JOB_READONLY.fullmatch(cmd)) is ok


def test_approval_card_explains_a_job_submit():
    assert "2 GPU(s), picked automatically" in _job_note("/x/deploy/sac-job submit --name r --gpus 2 -- python a.py")
    assert "CPU only" in _job_note("sac-job submit --name r -- python a.py")
    assert _job_note("python a.py") == "" and _job_note("sac-job list") == ""


def test_home_tab_lists_jobs(tmp_path: Path, cli):
    from slack_agent_controller import home

    async def run():
        f = _front(tmp_path, cli)

        async def fake_run(*cmd: str, timeout: float = 15.0) -> str:
            return "0, 0, 0, 46068"

        f.hub._run = fake_run  # type: ignore[method-assign]
        make_job(cli, "c0ffee", state="running", started=time.time(), gpus=1)
        cli.update_status("c0ffee", gpus=[3])
        view = await home.build_home(f, "UOWNER")
        body = json.dumps(view)
        assert "*Jobs* (1 running or queued)" in body and "c0ffee" in body and "sac_job_log_hc0ffee" in body
    asyncio.run(run())


def test_config_has_jobs_dir_readable(tmp_path: Path):
    cfg = dataclasses.replace(make_config(tmp_path, [tmp_path / "x"]), jobs_dir=tmp_path / "jobs",
                              read_only_roots=(tmp_path / "jobs",))
    assert cfg.readable(tmp_path / "jobs" / "abc123" / "log")


def test_claude_is_told_about_sac_job_only_when_jobs_are_enabled(tmp_path: Path):
    from slack_agent_controller.hub import Hub
    on = make_config(tmp_path, [tmp_path], jobs_enabled=True)
    off = make_config(tmp_path, [tmp_path], jobs_enabled=False)
    s_on = Hub(on, FakeWeb()).create_session("D1", "1.0", tmp_path, "U")  # type: ignore[arg-type]
    s_off = Hub(off, FakeWeb()).create_session("D1", "2.0", tmp_path, "U")  # type: ignore[arg-type]
    assert "sac-job submit" in s_on._system_append()
    assert "sac-job" not in s_off._system_append()
