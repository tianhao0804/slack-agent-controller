"""Terminal mirror (offline): rendering, redaction, event flow, lock, hook script, installer."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from slack_agent_controller import policy
from slack_agent_controller import transcript_md as tm
from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.mirror import Mirror, _pid_start
from tests.helpers import FakeWeb, make_config

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / "deploy" / "sac_hook.py"


# -- redaction / protection ---------------------------------------------------
def test_redact_log_keeps_paths_and_identifiers():
    keep = [
        "/data/lab/alice/projects/demo/runs/09_hea13/POSCAR_Cu2O_ICSD_26963_x",
        "Zr1O2_ICSD_12345_POCC_P0-1xD0-1xE0-1xF0-1xG0-1xH0",
        "commit 3f9a1c2b7d4e5f60718293a4b5c6d7e8f9012345",
        "session 20b6a7fe-9bdd-4690-97bc-d26f55fc0273",
        "runs/09_hea13/AFLOW_POCC_Cv_J_per_mol_formula_K_at298",
    ]
    for s in keep:
        assert policy.redact_log(s) == s, s


def test_redact_log_catches_secrets():
    secrets = [
        # token-shaped fakes are assembled at run time so secret scanners do not flag the repository
        "AB" + "SK" + "QmVkcm9ja0FQSUtleS1hYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk=",
        "xo" + "xb-" + "1234567890-abcdefghijkl",
        "AK" + "IA" + "IOSFODNN7" + "EXAMPLE",
        "sk-" + "ant-" + "api03-" + "x" * 40,
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "blob 9fK2mQ7xL4pZ8vR1tY6wB3nC5hJ0dG2sA7eF4kM9qX1",
    ]
    for s in secrets:
        out = policy.redact_log(s)
        assert "<redacted" in out, s


def test_output_protected(tmp_path: Path):
    home = Path.home()
    cfg = make_config(tmp_path, roots=[tmp_path], sensitive=[home / ".ssh", home / ".aws"])
    assert policy.output_protected(cfg, "Read", {"file_path": str(home / ".ssh" / "id_rsa")}, tmp_path)
    assert policy.output_protected(cfg, "Read", {"file_path": str(tmp_path / ".env")}, tmp_path)
    assert policy.output_protected(cfg, "Bash", {"command": "cat ~/.aws/credentials"}, tmp_path)
    assert policy.output_protected(cfg, "Bash", {"command": "echo $AWS_BEARER_TOKEN_BEDROCK"}, tmp_path)
    assert policy.output_protected(cfg, "Grep", {"pattern": "x", "path": str(home)}, tmp_path)
    assert policy.output_protected(cfg, "Bash", {"command": "cat .env"}, tmp_path)  # protected file name
    assert policy.output_protected(cfg, "Bash", {"command": 'cd x && head "sub dir/id_rsa"'}, tmp_path)
    assert policy.output_protected(cfg, "Bash", {"command": "python3 run.py --cert=server.pem"}, tmp_path)
    assert not policy.output_protected(cfg, "Bash", {"command": "ls -la"}, tmp_path)
    assert not policy.output_protected(cfg, "Bash", {"command": "tail -n 5 run.log | grep E"}, tmp_path)
    assert not policy.output_protected(cfg, "Read", {"file_path": str(tmp_path / "a.py")}, tmp_path)


# -- rendering ----------------------------------------------------------------
def _ts(sec: int) -> str:
    return f"2026-11-15T10:00:{sec:02d}.000Z"


def _user(text: Any, sec: int, **kw: Any) -> dict:
    return {"type": "user", "timestamp": _ts(sec), "message": {"role": "user", "content": text}, **kw}


def _asst(blocks: list[dict], sec: int, mid: str = "m1", out_tok: int = 10) -> dict:
    return {"type": "assistant", "timestamp": _ts(sec), "message": {
        "id": mid, "model": "claude-test", "content": blocks,
        "usage": {"input_tokens": 5, "cache_read_input_tokens": 100, "output_tokens": out_tok}}}


def _result(tid: str, text: str, sec: int, err: bool = False) -> dict:
    return _user([{"type": "tool_result", "tool_use_id": tid, "content": text, "is_error": err}], sec)


def sample_entries(tmp: Path) -> list[dict]:
    long_out = "\n".join(f"line {i}" for i in range(1000))
    return [
        {"type": "queue-operation", "operation": "enqueue"},
        _user("Please check the run", 0, origin={"kind": "human"}),
        _asst([{"type": "thinking", "thinking": "I should look at the log", "signature": "s"}], 1, "m1"),
        _asst([{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "tail -n 1000 run.log",
                                                                          "description": "show log"}}], 2, "m1"),
        _result("t1", long_out, 3),
        _asst([{"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": str(tmp / ".env")}},
               {"type": "tool_use", "id": "t3", "name": "Edit", "input": {
                   "file_path": str(tmp / "a.py"), "old_string": "x = 1", "new_string": "x = 2"}}], 4, "m2"),
        _result("t2", "SECRET_TOKEN_VALUE_should_not_appear", 5),
        _result("t3", "ok ``` with fence", 6),
        _user("<task-notification>bg done</task-notification>", 7, origin={"kind": "task-notification"}),
        _user("<command-name>/model</command-name>", 8),
        _asst([{"type": "tool_use", "id": "t4", "name": "Agent", "input": {"description": "dig", "prompt": "go"}}],
              9, "m3"),
        _result("t4", "agent says hi", 10),
        _asst([{"type": "text", "text": "All good: **done**."}], 11, "m4", out_tok=7),
        {"type": "system", "subtype": "turn_duration", "durationMs": 11000, "timestamp": _ts(11)},
    ]


def test_render_full_turn(tmp_path: Path):
    cfg = make_config(tmp_path, roots=[tmp_path])
    sub = tm.SubagentChunk("abc", "dig", "t4", True, [
        _user("go", 9),
        _asst([{"type": "tool_use", "id": "s1", "name": "Bash", "input": {"command": "ls"}}], 9, "sm1"),
        _result("s1", "file1\nfile2", 9),
        _asst([{"type": "text", "text": "found 2 files"}], 10, "sm2"),
    ])
    r = tm.render(cfg, tmp_path, "Terminal turn 1", sample_entries(tmp_path), [sub])
    md = r.markdown
    assert r.prompts == ["Please check the run"]
    assert r.final_text == "All good: **done**."
    assert r.tool_calls == 4 and r.subagents == 1
    assert r.tokens_out == (10 + 10 + 10 + 7) + (10 + 10)  # once per API message id; sub-agents included
    assert r.duration_s == 11
    assert "💭 *thinking*" in md and "I should look at the log" in md
    assert "tail -n 1000 run.log" in md and "line 0" in md and "line 999" in md
    assert "lines omitted" in md and "line 500\n" not in md          # head + tail kept
    assert "SECRET_TOKEN_VALUE" not in md and "omitted: touches a protected path" in md
    assert "-x = 1" in md and "+x = 2" in md                          # Edit rendered as diff
    assert "````\nok ``` with fence" in md                           # fence longer than content's
    assert "Background task notification" in md and "Inserted by Claude Code (command-name)" in md
    assert "Sub-agent: dig (launched by #4)" in md and "found 2 files" in md and "#a1.1" in md


def test_render_capped_stays_under_limit(tmp_path: Path):
    cfg = make_config(tmp_path, roots=[tmp_path])
    big = "x" * 50_000
    entries = [_user("go", 0)]
    for i in range(80):
        entries.append(_asst([{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": "cat f"}}],
                             1, f"m{i}"))
        entries.append(_result(f"t{i}", big, 2))
    r = tm.render_capped(cfg, tmp_path, "t", entries)
    assert len(r.markdown.encode()) <= tm.MAX_FILE_BYTES


def test_read_new_leaves_partial_line(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"type":"user"}\n{"type":"assis')
    entries, off = tm.read_new(p, 0)
    assert len(entries) == 1 and off == len('{"type":"user"}\n')
    with p.open("a") as f:
        f.write('tant"}\n')
    entries, off2 = tm.read_new(p, off)
    assert [e["type"] for e in entries] == ["assistant"] and off2 == p.stat().st_size


# -- event flow ---------------------------------------------------------------
class Env:
    def __init__(self, tmp: Path) -> None:
        self.proj = tmp / "proj"
        self.proj.mkdir()
        (tmp / "other").mkdir()
        self.cfg = make_config(tmp, roots=[tmp], allowed_channels=frozenset({"CMIR"}),
                               mirror_dirs=(self.proj.resolve(),), mirror_channel="CMIR", mirror_user="UOWNER",
                               mirror_socket=tmp / "m.sock")
        self.front = SlackFrontend(self.cfg)
        self.web = FakeWeb()
        self.front.hub.web = self.web  # type: ignore[assignment]
        self.hub = self.front.hub
        self.mirror = Mirror(self.hub)
        self.hub.mirror = self.mirror
        self.projects = tmp / "projects"
        self.mirror._projects = self.projects.resolve()
        (self.projects / "slug").mkdir(parents=True)
        self.sid = str(uuid.uuid4())
        self.tp = self.projects / "slug" / f"{self.sid}.jsonl"

    def ev(self, name: str, **kw: Any) -> dict:
        base = {"hook_event_name": name, "session_id": self.sid, "transcript_path": str(self.tp),
                "cwd": str(self.proj), "claude_pid": os.getpid(), "claude_pid_start": _pid_start(os.getpid())}
        return {**base, **kw}

    def append(self, entries: list[dict]) -> None:
        with self.tp.open("a") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def uploads(self) -> list[dict]:
        return [kw for op, kw in self.web.calls if op == "upload"]


def _fast(monkeypatch) -> None:
    import slack_agent_controller.mirror as mm
    monkeypatch.setattr(mm, "STOP_QUIET_S", 0.05)
    monkeypatch.setattr(mm, "STOP_WAIT_S", 1.0)
    monkeypatch.setattr(mm, "PROGRESS_INTERVAL_S", 0.05)


def test_full_session_flow(tmp_path: Path, monkeypatch):
    _fast(monkeypatch)

    async def run():
        e = Env(tmp_path)
        m = e.mirror
        e.append([{"type": "permission-mode", "permissionMode": "default"}])
        await m.handle(e.ev("SessionStart", source="startup"))
        assert not e.web.messages, "no thread before the first prompt"
        assert m.locked(e.sid) is not None
        await m.handle(e.ev("UserPromptSubmit", prompt="Please check the run"))
        root = next(ts for ts, msg in e.web.messages.items() if msg.get("thread_ts") is None)
        rec = e.hub.store.find_by_session(e.sid)
        assert rec is not None and rec.thread_ts == root and rec.channel == "CMIR" and rec.has_history
        await m.handle(e.ev("PreToolUse", tool_name="Bash", tool_input={"command": "tail -n 1000 run.log"}))
        await asyncio.sleep(0.2)
        assert any("tail -n 1000 run.log" in t for t in e.web.texts()), "progress shows the tool call"

        # Slack prompt while the terminal has the session open: refused.
        await e.front._prompt("CMIR", "9.0", root, "UOWNER", "hello from slack")
        assert any("open in a terminal" in t for t in e.web.texts())
        assert e.hub.sessions[rec.key].worker is None, "nothing was sent to Claude"

        e.append(sample_entries(e.proj))
        await m.handle(e.ev("Stop", last_assistant_message="All good"))
        ups = e.uploads()
        assert len(ups) == 1 and ups[0]["filename"] == "turn-001.md" and ups[0]["thread_ts"] == root
        assert "tail -n 1000 run.log" in ups[0]["content"] and "SECRET_TOKEN_VALUE" not in ups[0]["content"]
        assert any("Terminal turn 1" in t and "All good" in t for t in e.web.texts())

        # Nothing new -> no second file.
        await m.handle(e.ev("Stop"))
        assert len(e.uploads()) == 1

        await m.handle(e.ev("SessionEnd", reason="prompt_input_exit"))
        assert m.locked(e.sid) is None
        assert any("Terminal closed" in t for t in e.web.texts())
        assert "Terminal closed" in e.web.messages[root]["text"], "root updated"
        assert m.states[e.sid].turns == 1 and m.states[e.sid].offset == e.tp.stat().st_size

        # State survives a restart.
        m2 = Mirror(e.hub)
        assert m2.states[e.sid].thread_ts == root and m2.states[e.sid].turns == 1

    asyncio.run(run())


def test_resume_skips_old_history_and_reuses_thread(tmp_path: Path, monkeypatch):
    _fast(monkeypatch)

    async def run():
        e = Env(tmp_path)
        e.append(sample_entries(e.proj))  # history written before (e.g. by Slack or an older terminal run)
        e.hub.create_session("CMIR", "5.0", e.proj, "UOWNER", session_id=e.sid)  # thread already exists
        await e.mirror.handle(e.ev("SessionStart", source="resume"))
        assert any("Opened in a terminal" in m.get("text", "") and m.get("thread_ts") == "5.0"
                   for m in e.web.messages.values())
        await e.mirror.handle(e.ev("UserPromptSubmit", prompt="new question"))
        e.append([_user("new question", 30, origin={"kind": "human"}),
                  _asst([{"type": "text", "text": "new answer"}], 31, "n1")])
        await e.mirror.handle(e.ev("Stop"))
        content = e.uploads()[0]["content"]
        assert "new answer" in content and "Please check the run" not in content
        assert all(kw.get("thread_ts") in ("5.0", None) for _, kw in e.web.calls if "thread_ts" in kw)

    asyncio.run(run())


def test_unmirrored_dir_and_bad_path_are_ignored(tmp_path: Path):
    async def run():
        e = Env(tmp_path)
        await e.mirror.handle(e.ev("UserPromptSubmit", prompt="x", cwd=str(tmp_path / "other")))
        bad = {**e.ev("UserPromptSubmit", prompt="x"), "session_id": str(uuid.uuid4())}
        await e.mirror.handle(bad)  # transcript name does not match its session id
        await e.mirror.handle({**e.ev("UserPromptSubmit", prompt="x"), "session_id": "../../etc"})
        assert not e.web.calls and not e.mirror.states

    asyncio.run(run())


def test_dead_terminal_process_unlocks(tmp_path: Path):
    async def run():
        e = Env(tmp_path)
        p = subprocess.Popen(["true"])
        p.wait()
        await e.mirror.handle(e.ev("SessionStart", source="startup", claude_pid=p.pid, claude_pid_start=12345))
        assert e.mirror.states[e.sid].open
        assert e.mirror.locked(e.sid) is None and not e.mirror.states[e.sid].open

    asyncio.run(run())


def test_permission_notification_mentions_owner(tmp_path: Path):
    async def run():
        e = Env(tmp_path)
        await e.mirror.handle(e.ev("Notification", notification_type="idle_prompt", message="waiting"))
        assert not e.web.calls
        await e.mirror.handle(e.ev("Notification", notification_type="permission_prompt",
                                   message="Claude needs your permission to use Bash"))
        assert any("<@UOWNER>" in t and "permission" in t for t in e.web.texts())

    asyncio.run(run())


def test_deleting_root_mutes_until_reopened(tmp_path: Path, monkeypatch):
    _fast(monkeypatch)

    async def run():
        e = Env(tmp_path)
        await e.mirror.handle(e.ev("UserPromptSubmit", prompt="hi"))
        rec = e.hub.store.find_by_session(e.sid)
        await e.front._end_session(rec.channel, rec.thread_ts, e.sid, rec.cwd, "UOWNER")
        n = len(e.web.calls)
        await e.mirror.handle(e.ev("PreToolUse", tool_name="Bash", tool_input={"command": "ls"}))
        assert len(e.web.calls) == n, "muted: nothing posted"
        await e.mirror.handle(e.ev("SessionStart", source="resume"))
        await e.mirror.handle(e.ev("UserPromptSubmit", prompt="again"))
        assert e.hub.store.find_by_session(e.sid) is not None, "new thread after reopening"

    asyncio.run(run())


# -- hook script and socket ---------------------------------------------------
def test_hook_script_is_silent_without_bot(tmp_path: Path):
    payload = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "x", "prompt": "hi"})
    out = subprocess.run([sys.executable, "-I", str(HOOK), "--socket", str(tmp_path / "none.sock")],
                         input=payload, capture_output=True, text=True, timeout=10)
    assert out.returncode == 0 and out.stdout == ""


def test_hook_to_socket_roundtrip(tmp_path: Path):
    async def run():
        e = Env(tmp_path)
        got: list[dict] = []

        async def record(ev: dict) -> None:
            got.append(ev)

        e.mirror.handle = record  # type: ignore[method-assign]
        await e.mirror.start()
        assert oct(e.cfg.mirror_socket.stat().st_mode & 0o777) == "0o600"
        payload = {"hook_event_name": "PreToolUse", "session_id": e.sid, "tool_name": "Write",
                   "tool_input": {"file_path": "a", "content": "y" * 10_000}, "tool_response": "dropped"}
        env = {**os.environ, "CLAUDE_PID": str(os.getpid())}
        out = await asyncio.to_thread(subprocess.run, [sys.executable, "-I", str(HOOK), "--socket",
                                                       str(e.cfg.mirror_socket)],
                                      input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=10)
        for _ in range(50):
            if got:
                break
            await asyncio.sleep(0.05)
        await e.mirror.stop()
        assert out.returncode == 0 and out.stdout == ""
        ev = got[0]
        assert ev["tool_name"] == "Write" and "tool_response" not in ev
        assert len(ev["tool_input"]["content"]) < 4100 and ev["claude_pid"] == os.getpid()
        assert ev["claude_pid_start"] == _pid_start(os.getpid())

    asyncio.run(run())


def test_installer_is_idempotent_and_reversible(tmp_path: Path):
    s = tmp_path / "settings.json"
    other = {"type": "command", "command": "echo mine"}
    original = {"model": "opus", "hooks": {"Stop": [{"hooks": [other]}]}}
    s.write_text(json.dumps(original))
    inst = [sys.executable, str(ROOT / "deploy" / "install_hooks.py"), "--settings", str(s)]
    for _ in range(2):
        subprocess.run(inst, check=True, capture_output=True)
    d = json.loads(s.read_text())
    assert d["model"] == "opus"
    assert sum("sac_hook.py" in h["command"] for g in d["hooks"]["Stop"] for h in g["hooks"]) == 1
    assert {"hooks": [other]} in d["hooks"]["Stop"]
    assert d["hooks"]["PreToolUse"][0]["matcher"] == "*"
    subprocess.run(inst + ["--remove"], check=True, capture_output=True)
    assert json.loads(s.read_text()) == original
    assert len(list(tmp_path.glob("settings.json.bak-*"))) >= 1


def test_config_validation(tmp_path: Path):
    cfg = make_config(tmp_path, roots=[tmp_path])
    assert not cfg.mirror_dirs and not cfg.mirrored(tmp_path)
    cfg2 = dataclasses.replace(cfg, mirror_dirs=(tmp_path,))
    assert cfg2.mirrored(tmp_path / "x")


def test_background_session_lock_and_unlock(tmp_path: Path, monkeypatch):
    import slack_agent_controller.mirror as mm
    monkeypatch.setattr(mm, "_is_background", lambda pid: True)

    async def run():
        e = Env(tmp_path)
        await e.mirror.handle(e.ev("UserPromptSubmit", prompt="hi"))
        st = e.mirror.locked(e.sid)
        assert st is not None and st.background
        rec = e.hub.store.find_by_session(e.sid)
        await e.front._prompt(rec.channel, "9.0", rec.thread_ts, "UOWNER", "hello")
        assert any("background session" in t and f"claude stop {e.sid[:8]}" in t for t in e.web.texts())
        # !unlock is refused while the process lives; "force" overrides.
        await e.front._cmd_unlock(rec.channel, "9.1", rec.thread_ts, "UOWNER", "")
        assert e.mirror.locked(e.sid) is not None
        assert any("Not unlocked" in t for t in e.web.texts())
        await e.front._cmd_unlock(rec.channel, "9.2", rec.thread_ts, "UOWNER", "force")
        assert e.mirror.locked(e.sid) is None

    asyncio.run(run())


def test_background_cli_error_is_explained(tmp_path: Path):
    hub = SlackFrontend(make_config(tmp_path, roots=[tmp_path])).hub
    sess = hub.create_session("D1", "1.0", tmp_path, "UOWNER")
    sess._stderr.append("Error: That session is running in the background (c9fcb0df). Run claude attach c9fcb0df")
    msg = sess._error_text(RuntimeError("Command failed with exit code 1"))
    assert "claude stop c9fcb0df" in msg and "claude attach c9fcb0df" in msg


def test_is_background_on_real_processes():
    from slack_agent_controller.mirror import _is_background
    assert not _is_background(os.getpid())
    assert not _is_background(999_999_999)


def test_model_effort_commands(tmp_path: Path):
    async def run():
        f = SlackFrontend(make_config(tmp_path, roots=[tmp_path], model="sonnet"))
        web = FakeWeb()
        f.hub.web = web  # type: ignore[assignment]
        sess = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        assert sess._options().effort is None and sess._options().model == "sonnet"
        await f._cmd_effort("D1", "2.0", "1.0", "UOWNER", "")
        assert "Claude Code default" in web.texts()[-1]
        await f._cmd_effort("D1", "2.1", "1.0", "UOWNER", "turbo")
        assert "Usage" in web.texts()[-1]
        await f._cmd_effort("D1", "2.2", "1.0", "UOWNER", "high")
        await f._cmd_model("D1", "2.3", "1.0", "UOWNER", "opus")
        o = sess._options()
        assert o.effort == "high" and o.model == "opus"
        assert f.hub.store.get(sess.rec.key).effort == "high"  # persisted
        await f._cmd_model("D1", "2.4", "1.0", "UOWNER", "gpt-4; rm -rf /")
        assert "Usage" in web.texts()[-1] and sess._options().model == "opus"
        await f._cmd_model("D1", "2.5", "1.0", "UOWNER", "default")
        assert sess._options().model == "sonnet"

        class FakeClient:
            async def disconnect(self) -> None:
                pass

        sess.client = FakeClient()  # type: ignore[assignment]
        assert "restarts" in sess.set_model_effort(effort="max") and sess._options_stale
        await sess._disconnect()
        assert not sess._options_stale

    asyncio.run(run())


def test_mirror_carries_terminal_model_and_effort(tmp_path: Path, monkeypatch):
    _fast(monkeypatch)

    async def run():
        e = Env(tmp_path)
        await e.mirror.handle(e.ev("UserPromptSubmit", prompt="q"))
        reply = _asst([{"type": "text", "text": "a"}], 1, "x1")
        reply.update(entrypoint="cli", requestedModel="us.anthropic.claude-opus-5-5", effort="high")
        e.append([_user("q", 0, origin={"kind": "human"}), reply])
        await e.mirror.handle(e.ev("Stop"))
        rec = e.hub.store.find_by_session(e.sid)
        assert rec.model == "opus" and rec.effort == "high"

    asyncio.run(run())


def test_exit_command(tmp_path: Path):
    class FakeClient:
        def __init__(self) -> None:
            self.disconnected = False

        async def disconnect(self) -> None:
            self.disconnected = True

    async def run():
        f = SlackFrontend(make_config(tmp_path, roots=[tmp_path]))
        web = FakeWeb()
        f.hub.web = web  # type: ignore[assignment]
        await f._cmd_exit("D1", "0.1", None, "UOWNER", "")
        assert "inside a session thread" in web.texts()[-1]
        sess = f.hub.create_session("D1", "1.0", tmp_path, "UOWNER")
        await f._cmd_exit("D1", "0.2", "1.0", "UOWNER", "")
        assert "Nothing to close" in web.texts()[-1] and sess.queue.empty()

        def connect() -> FakeClient:
            c = FakeClient()
            sess.client = c  # type: ignore[assignment]
            sess.worker = asyncio.create_task(sess._worker())
            return c

        c = connect()
        sess.bg_tasks["t"] = "watch job"
        await f._cmd_exit("D1", "0.3", "1.0", "UOWNER", "")
        assert "Background task" in web.texts()[-1] and not c.disconnected
        await f._cmd_exit("D1", "0.4", "1.0", "UOWNER", "force")
        assert c.disconnected and sess.client is None and sess.worker.done() and sess.queue.empty()
        assert "claude --resume" in web.texts()[-1] and "1 background task" in web.texts()[-1]

        c = connect()
        await f._cmd_exit("D1", "0.5", "1.0", "UOWNER", "")
        assert c.disconnected and "Closed this thread" in web.texts()[-1]
        assert f.hub.store.get(sess.rec.key) is not None, "mapping kept"
        assert '"session-exit"' in (f.cfg.state_dir / "audit.jsonl").read_text()

    asyncio.run(run())
