"""Files attached in Slack (review R9): download, limits, safety, receipt, prompt."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

import slack_agent_controller.inbox as inbox_mod
from slack_agent_controller import policy
from slack_agent_controller.app import SlackFrontend
from slack_agent_controller.inbox import (
    Inbox,
    kind_of,
    prompt_with_files,
    safe_name,
    url_ok,
)
from tests.helpers import FakeWeb, make_config

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 100


def _cfg(tmp: Path, **over: Any):
    box = tmp / "inbox"
    cfg = make_config(tmp, [tmp / "proj"], inbox_dir=box, read_only_roots=(box,), **over)
    (tmp / "proj").mkdir(exist_ok=True)
    return cfg


# -- pure helpers ---------------------------------------------------------------
@pytest.mark.parametrize("f,kind", [
    ({"name": "a.png", "filetype": "png", "mimetype": "image/png"}, "image"),
    ({"name": "shot.JPG", "filetype": "jpg"}, "image"),
    ({"name": "paper.pdf", "mimetype": "application/pdf"}, "pdf"),
    ({"name": "run.log", "filetype": "text", "mimetype": "text/plain"}, "text"),
    ({"name": "OUTCAR", "filetype": "outcar"}, "text"),
    ({"name": "x.ipynb"}, "notebook"),
    ({"name": "snippet", "mode": "snippet"}, "text"),
    ({"name": "data.h5", "filetype": "binary", "mimetype": "application/octet-stream"}, None),
    ({"name": "movie.mp4", "mimetype": "video/mp4"}, None),
])
def test_kind_of(f, kind):
    assert kind_of(f) == kind


@pytest.mark.parametrize("raw,safe", [
    ("../../etc/passwd", "passwd"), ("..", "file"), (".hidden", "hidden"), ("a b/c d.txt", "c_d.txt"),
    ("screen shot 2026-11-17 at 10.00.png", "screen_shot_2026-11-17_at_10.00.png"), ("日志.txt", "_.txt"),
    ("x" * 300 + ".log", "x" * 96 + ".log"), ("C:\\Users\\me\\a.txt", "a.txt"),
])
def test_safe_name(raw, safe):
    assert safe_name(raw) == safe


def test_url_ok():
    assert url_ok("https://files.slack.com/files-pri/T1-F1/download/a.png")
    for bad in ("http://files.slack.com/x", "https://evil.example/x", "https://files.slack.com.evil.example/x",
                "https://user@evil.example/x", "file:///etc/passwd", ""):
        assert not url_ok(bad), bad


def test_prompt_with_files(tmp_path: Path):
    s = inbox_mod.Saved(name="a.png", path=tmp_path / "a.png", kind="image", size=12000)
    p = prompt_with_files("why does this fail?", [s])
    assert p.startswith("why does this fail?") and str(tmp_path / "a.png") in p and "Read tool" in p
    assert "only sent the file" in prompt_with_files("", [s])


def test_inbox_is_readable_by_policy_without_approval(tmp_path: Path):
    cfg = _cfg(tmp_path)
    f = cfg.inbox_dir / "abcd1234" / "x.png"
    assert policy.decide(cfg, "Read", {"file_path": str(f)}, tmp_path / "proj").decision == "allow"


def test_cleanup_removes_only_old_files(tmp_path: Path):
    box = Inbox(_cfg(tmp_path))
    d = box.session_dir("abcdef123456")
    old, new = d / "old.txt", d / "new.txt"
    old.write_text("x")
    new.write_text("y")
    past = time.time() - 8 * 86400
    os.utime(old, (past, past))
    assert box.cleanup() == 1 and not old.exists() and new.exists()
    new.unlink()
    box.cleanup()
    assert not d.exists(), "empty session folder removed"


# -- downloads against a local stand-in for files.slack.com ---------------------
class Server:
    def __init__(self) -> None:
        self.auth: list[str] = []
        self.app = web.Application()
        self.app.router.add_get("/png", self.png)
        self.app.router.add_get("/text", self.text)
        self.app.router.add_get("/redirect", self.redirect)
        self.app.router.add_get("/login", self.login)
        self.app.router.add_get("/huge", self.huge)

    async def png(self, req):
        self.auth.append(req.headers.get("Authorization", ""))
        return web.Response(body=PNG, content_type="image/png")

    async def text(self, req):
        self.auth.append(req.headers.get("Authorization", ""))
        return web.Response(text="line 1\nERROR: boom\n", content_type="text/plain")

    async def redirect(self, req):
        raise web.HTTPFound("https://slack.com/signin")

    async def login(self, req):
        return web.Response(text="<html>sign in</html>", content_type="text/html")

    async def huge(self, req):
        return web.Response(body=b"x" * 3_000_000, content_type="text/plain")

    async def __aenter__(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[union-attr]
        return self

    async def __aexit__(self, *a):
        await self.runner.cleanup()


@pytest.fixture
def local_slack(monkeypatch):
    monkeypatch.setattr(inbox_mod, "ALLOWED_HOSTS", frozenset({"127.0.0.1"}))
    monkeypatch.setattr(inbox_mod, "ALLOWED_SCHEMES", frozenset({"http"}))


def _file(srv: Server, path: str, name: str, **kw: Any) -> dict:
    return {"id": "F" + name, "name": name, "url_private_download": srv.base + path, "size": 100, **kw}


def test_fetch_saves_files_privately_with_the_token(tmp_path: Path, local_slack):
    async def run():
        cfg = _cfg(tmp_path)
        box = Inbox(cfg)
        async with Server() as srv:
            saved, problems = await box.fetch([
                _file(srv, "/png", "error shot.png", filetype="png", mimetype="image/png"),
                _file(srv, "/text", "run.log", filetype="text", mimetype="text/plain"),
            ], "0123456789ab-session", FakeWeb())
        assert not problems and [s.kind for s in saved] == ["image", "text"]
        assert saved[0].path.read_bytes() == PNG and saved[0].path.name.endswith("-error_shot.png")
        assert saved[0].path.parent == cfg.inbox_dir / "01234567"
        assert stat.S_IMODE(saved[0].path.stat().st_mode) == 0o600
        assert stat.S_IMODE(saved[0].path.parent.stat().st_mode) == 0o700
        assert srv.auth == ["Bearer xoxb-test", "Bearer xoxb-test"]
    asyncio.run(run())


def test_fetch_reports_problems(tmp_path: Path, local_slack):
    async def run():
        cfg = dataclasses.replace(_cfg(tmp_path), inbox_max_mb=1.0)
        box = Inbox(cfg)
        async with Server() as srv:
            saved, problems = await box.fetch([
                _file(srv, "/png", "data.h5", filetype="binary", mimetype="application/octet-stream"),
                _file(srv, "/png", "big.png", filetype="png", size=5_000_000),
                _file(srv, "/huge", "lies.log", filetype="text", size=10),       # declared small, is 3 MB
                _file(srv, "/text", ".env", filetype="text"),
                _file(srv, "/text", "server.pem", filetype="text"),
                {"id": "F9", "name": "x.png", "filetype": "png", "url_private_download": "https://evil.example/x"},
            ], "abcdefgh", FakeWeb())
        assert not saved
        text = " | ".join(problems)
        assert "data.h5" in text and "not supported" in text
        assert "big.png" in text and "limit" in text
        assert "lies.log" in text and "download failed" in text
        assert ".env" in text and "server.pem" in text and "secret" in text
        assert "x.png" in text
        assert not list(cfg.inbox_dir.rglob("*lies*")), "partial download removed"
    asyncio.run(run())


@pytest.mark.parametrize("path", ["/redirect", "/login"])
def test_missing_files_read_scope_is_explained(tmp_path: Path, local_slack, path):
    async def run():
        box = Inbox(_cfg(tmp_path))
        async with Server() as srv:
            saved, problems = await box.fetch([_file(srv, path, "a.png", filetype="png"),
                                               _file(srv, path, "b.png", filetype="png")], "abcdefgh", FakeWeb())
        assert not saved and len(problems) == 1 and "files:read" in problems[0]
    asyncio.run(run())


def test_files_info_is_used_when_the_event_has_no_url(tmp_path: Path, local_slack):
    async def run():
        box = Inbox(_cfg(tmp_path))
        async with Server() as srv:
            class Web(FakeWeb):
                async def files_info(self, **kw):
                    assert kw["file"] == "F1"
                    return {"file": _file(srv, "/png", "a.png", filetype="png")}

            saved, problems = await box.fetch([{"id": "F1", "file_access": "check_file_info"}], "abcdefgh", Web())
        assert not problems and saved[0].path.read_bytes() == PNG
        assert saved[0].name == "a.png", "the receipt shows the file name, not its id"
    asyncio.run(run())


def test_too_many_files(tmp_path: Path, local_slack):
    async def run():
        box = Inbox(_cfg(tmp_path))
        async with Server() as srv:
            files = [_file(srv, "/text", f"f{i}.log", filetype="text") for i in range(12)]
            saved, problems = await box.fetch(files, "abcdefgh", FakeWeb())
        assert len(saved) == inbox_mod.MAX_FILES and "first 10 of 12" in problems[0]
    asyncio.run(run())


# -- the Slack side ---------------------------------------------------------------
def test_file_message_gets_a_receipt_and_reaches_claude(tmp_path: Path, local_slack):
    async def run():
        f = SlackFrontend(_cfg(tmp_path))
        fake = FakeWeb()
        f.hub.web = fake  # type: ignore[assignment]
        sent: list[str] = []
        async with Server() as srv:
            def capture(self, text, user):  # noqa: ANN001
                sent.append(text)
                return 0
            import types
            orig = f.hub.create_session

            def create(*a, **k):  # noqa: ANN002, ANN003
                s = orig(*a, **k)
                s.submit = types.MethodType(capture, s)  # type: ignore[method-assign]
                return s

            f.hub.create_session = create  # type: ignore[method-assign]
            event = {"type": "message", "subtype": "file_share", "channel_type": "im", "channel": "D1",
                     "user": "UOWNER", "ts": "7.0", "text": "",
                     "files": [_file(srv, "/png", "shot.png", filetype="png", mimetype="image/png"),
                               _file(srv, "/png", "data.h5", filetype="binary")]}
            await f._on_message(event)
            for t in list(f._tasks):
                await t
        assert len(sent) == 1 and "only sent the file" in sent[0] and "shot.png" in sent[0]
        texts = fake.texts()
        assert any("Received: `shot.png` (image" in t and "data.h5" in t for t in texts), texts
        audit = (f.cfg.state_dir / "audit.jsonl").read_text()
        assert '"event": "inbox"' in audit
    asyncio.run(run())


def test_nothing_usable_and_no_text_sends_nothing(tmp_path: Path, local_slack):
    async def run():
        f = SlackFrontend(_cfg(tmp_path))
        f.hub.web = FakeWeb()  # type: ignore[assignment]
        sess = f.hub.create_session("D1", "1.0", tmp_path / "proj", "UOWNER")
        text = await f._receive_files(sess, [{"id": "F", "name": "a.mp4", "mimetype": "video/mp4"}], "", "UOWNER")
        assert text == "" and any("Nothing was sent" in t for t in f.hub.web.texts())  # type: ignore[attr-defined]
    asyncio.run(run())


def test_missing_scope_on_files_info_is_explained(tmp_path: Path):
    from slack_sdk.errors import SlackApiError
    from slack_sdk.web.slack_response import SlackResponse

    class Web(FakeWeb):
        async def files_info(self, **kw):
            resp = SlackResponse(client=None, http_verb="GET", api_url="files.info", req_args={},
                                 data={"ok": False, "error": "missing_scope", "needed": "files:read"},
                                 headers={}, status_code=200)
            raise SlackApiError("missing_scope", resp)

    async def run():
        saved, problems = await Inbox(_cfg(tmp_path)).fetch([{"id": "F1", "file_access": "check_file_info"}],
                                                            "abcdefgh", Web())
        assert not saved and "files:read" in problems[0]
    asyncio.run(run())
