"""Files you attach to a Slack message (screenshots, logs, PDFs) -> local files Claude can Read.

Files are downloaded with the bot token (needs the ``files:read`` scope) into
``INBOX_DIR/<session>/`` (default ``~/.cache/slack-agent-controller/inbox``), outside your
project directories. That directory is a read-only root for the policy, so
the Read tool can open them (it shows images and PDFs to Claude) without an
approval card. Files older than ``INBOX_KEEP_DAYS`` are deleted.

The token is only ever sent to ``files.slack.com`` over HTTPS and redirects
are not followed, so it cannot be forwarded to another host.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from slack_sdk.errors import SlackApiError

from . import policy
from .config import Config

log = logging.getLogger(__name__)

ALLOWED_HOSTS = frozenset({"files.slack.com"})
ALLOWED_SCHEMES = frozenset({"https"})
MAX_FILES = 10
DOWNLOAD_TIMEOUT_S = 120.0

_IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "webp"}
_TEXT_TYPES = {
    "text", "txt", "log", "out", "err", "csv", "tsv", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf",
    "md", "markdown", "rst", "tex", "bib", "py", "sh", "bash", "zsh", "c", "h", "cpp", "hpp", "cc", "f", "f90",
    "f95", "jl", "r", "m", "js", "ts", "java", "go", "rs", "sql", "xml", "html", "css", "diff", "patch",
    "dat", "xyz", "cif", "poscar", "incar", "kpoints", "outcar", "contcar", "slurm", "sbatch", "lammps", "in",
}


@dataclass
class Saved:
    name: str        # as shown in Slack
    path: Path
    kind: str        # "image" | "pdf" | "text" | "notebook"
    size: int


def kind_of(f: dict[str, Any]) -> str | None:
    """What Claude can do with the file: image / pdf / text / notebook, or None (unsupported)."""
    ftype = str(f.get("filetype") or "").lower()
    mime = str(f.get("mimetype") or "").lower()
    ext = Path(str(f.get("name") or "")).suffix.lower().lstrip(".")
    if ftype in _IMAGE_TYPES or mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        return "image"
    if ftype == "pdf" or mime == "application/pdf" or ext == "pdf":
        return "pdf"
    if ext == "ipynb" or ftype == "ipynb":
        return "notebook"
    if (mime.startswith("text/") or mime in ("application/json", "application/x-yaml", "application/xml")
            or ftype in _TEXT_TYPES or ext in _TEXT_TYPES or str(f.get("mode")) == "snippet"):
        return "text"
    return None


def safe_name(name: str) -> str:
    """A plain file name: no directories, no leading dots, only [A-Za-z0-9._-], at most 100 chars."""
    base = os.path.basename(name.replace("\\", "/")).strip() or "file"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip(".") or "file"
    stem, dot, ext = base.rpartition(".")
    if dot and len(ext) <= 10:
        return stem[: 100 - len(ext) - 1] + "." + ext if stem else "file." + ext
    return base[:100]


def url_ok(url: str) -> bool:
    u = urlparse(url)
    return u.scheme in ALLOWED_SCHEMES and (u.hostname or "") in ALLOWED_HOSTS


class MissingScope(Exception):
    """Slack answered with a login page or a redirect: the app lacks files:read."""


class Inbox:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.root = cfg.inbox_dir

    def session_dir(self, session_id: str) -> Path:
        d = self.root / session_id[:8]
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(d, 0o700)
        return d

    def cleanup(self, now: float | None = None) -> int:
        """Delete files older than INBOX_KEEP_DAYS and empty session folders."""
        if not self.root.is_dir():
            return 0
        cutoff = (now or time.time()) - self.cfg.inbox_keep_days * 86400
        removed = 0
        for d in self.root.iterdir():
            if not d.is_dir() or d.is_symlink():
                continue
            for f in d.iterdir():
                try:
                    if f.is_file() and not f.is_symlink() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
            try:
                d.rmdir()  # only succeeds when empty
            except OSError:
                pass
        return removed

    async def fetch(self, files: list[dict[str, Any]], session_id: str, web: Any) -> tuple[list[Saved], list[str]]:
        """Download the message's files. Returns (saved, problems as short sentences)."""
        saved: list[Saved] = []
        problems: list[str] = []
        limit = int(self.cfg.inbox_max_mb * 1_000_000)
        if len(files) > MAX_FILES:
            problems.append(f"only the first {MAX_FILES} of {len(files)} files are used")
            files = files[:MAX_FILES]
        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            for f in files:
                name = str(f.get("name") or f.get("title") or f.get("id") or "file")
                try:
                    if f.get("file_access") == "check_file_info" or not f.get("url_private_download"):
                        try:
                            info = await web.files_info(file=f.get("id"))
                        except SlackApiError as e:
                            if (e.response.get("error") if e.response else "") in ("missing_scope", "not_allowed_token_type"):
                                raise MissingScope from e
                            raise
                        f = dict(info.get("file") or {})
                        name = str(f.get("name") or f.get("title") or name)
                    kind = kind_of(f)
                    size = int(f.get("size") or 0)
                    if f.get("mode") in ("tombstone", "hidden_by_limit") or f.get("is_external"):
                        problems.append(f"`{name}`: not available to the bot")
                    elif kind is None:
                        problems.append(f"`{name}`: type not supported (images, PDF, text files and notebooks are)")
                    elif size > limit:
                        problems.append(f"`{name}`: {size / 1e6:.1f} MB is over the {self.cfg.inbox_max_mb:.0f} MB limit")
                    elif self._protected(name):
                        problems.append(f"`{name}`: its name looks like a secret file, so Claude could not read it; "
                                        "rename it and send it again")
                    else:
                        path = await self._download(http, f, session_id, limit)
                        saved.append(Saved(name=name, path=path, kind=kind, size=path.stat().st_size))
                except MissingScope:
                    problems.append("the Slack app is missing the `files:read` permission: add it under OAuth & "
                                    "Permissions -> Bot Token Scopes and reinstall the app")
                    break
                except Exception as e:  # network, Slack API error, disk ...
                    log.warning("inbox: could not fetch %s: %s", name, type(e).__name__)
                    problems.append(f"`{name}`: download failed ({type(e).__name__})")
        return saved, problems

    def _protected(self, name: str) -> bool:
        return policy.is_sensitive(self.cfg, self.root / "x" / safe_name(name)) or \
            policy.is_sensitive(self.cfg, self.root / "x" / os.path.basename(name))

    async def _download(self, http: aiohttp.ClientSession, f: dict[str, Any], session_id: str, limit: int) -> Path:
        url = str(f.get("url_private_download") or f.get("url_private") or "")
        if not url_ok(url):
            raise ValueError("unexpected file URL")
        headers = {"Authorization": f"Bearer {self.cfg.slack_bot_token}"}
        async with http.get(url, headers=headers, allow_redirects=False) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if resp.status in (301, 302, 303, 307, 308) or (resp.status == 200 and ctype.startswith("text/html")
                                                             and kind_of(f) != "text"):
                raise MissingScope
            if resp.status in (401, 403):
                raise MissingScope
            resp.raise_for_status()
            d = self.session_dir(session_id)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            target = d / f"{stamp}-{safe_name(str(f.get('name') or 'file'))}"
            n = 1
            while target.exists():
                target = d / f"{stamp}-{n}-{safe_name(str(f.get('name') or 'file'))}"
                n += 1
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            total = 0
            try:
                with os.fdopen(fd, "wb") as out:
                    async for chunk in resp.content.iter_chunked(1 << 16):
                        total += len(chunk)
                        if total > limit:
                            raise ValueError("file grew past the size limit while downloading")
                        out.write(chunk)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            return target


def prompt_with_files(text: str, saved: list[Saved]) -> str:
    """The prompt Claude receives: your message plus where the files are."""
    lines = [f"- {s.path} ({s.kind}, {s.size / 1000:.0f} KB; sent as \"{s.name}\")" for s in saved]
    intro = text.strip() or ("(No message text: the user only sent the file(s) below. Look at them and, if what "
                             "they want is unclear, ask.)")
    return (f"{intro}\n\n[The user attached {len(saved)} file(s) in Slack. They are saved on this node; open them "
            "with the Read tool (it shows images and PDFs):\n" + "\n".join(lines) + "]")


async def cleanup_loop(inbox: Inbox) -> None:
    while True:
        try:
            n = await asyncio.to_thread(inbox.cleanup)
            if n:
                log.info("inbox: deleted %d file(s) older than %d days", n, inbox.cfg.inbox_keep_days)
        except Exception:
            log.exception("inbox cleanup failed")
        await asyncio.sleep(6 * 3600)
