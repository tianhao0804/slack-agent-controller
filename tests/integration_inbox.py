"""Integration: Claude reads an attached image from the inbox without an approval card
(real Bedrock/Haiku, fake Slack). The Slack download itself is covered offline and by a live check.

Run: .venv/bin/python -m tests.integration_inbox
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from slack_agent_controller.config import strip_parent_session_env
from slack_agent_controller.hub import Hub
from slack_agent_controller.inbox import Saved, prompt_with_files
from tests.helpers import FakeWeb, is_reply, make_config
from tests.integration_fake_slack import check, results, wait_idle

strip_parent_session_env()
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"


def solid_png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\0" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sac_inbox_"))
    proj = tmp / "proj"
    proj.mkdir()
    box = tmp / "inbox"
    cfg = make_config(tmp, roots=[proj], model="haiku", default_cwd=proj.resolve(), inbox_dir=box,
                      read_only_roots=(box.resolve(),))
    hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
    cards: list[str] = []

    async def deny(sess, tool, tin, ctx, reason):  # noqa: ANN001
        cards.append(f"{tool}: {tin}")
        return "deny", "UOWNER"

    hub.request_approval = deny  # type: ignore[method-assign]
    sess = hub.create_session("D1", "1.0", proj.resolve(), "UOWNER")
    img = hub.inbox.session_dir(sess.rec.session_id) / "20261117-120000-swatch.png"
    img.write_bytes(solid_png(64, 64, (220, 20, 20)))
    sess.submit(prompt_with_files("What single color fills this image? Answer with one word.",
                                  [Saved(name="swatch.png", path=img, kind="image", size=img.stat().st_size)]), "UOWNER")
    await wait_idle(sess, 240)
    audit = [json.loads(x) for x in (cfg.state_dir / "audit.jsonl").read_text().splitlines() if x.strip()]
    reads = [a for a in audit if a.get("tool") == "Read"]
    ans = [kw["text"] for op, kw in hub.web.calls if op == "update" and is_reply(kw)][-1]
    check("Claude opened the file with Read", bool(reads), str(audit[-3:]))
    check("Read allowed by policy (no card)", reads and all(a["decision"] == "allow" for a in reads) and not cards,
          str(cards))
    check("Claude saw the image", "red" in ans.lower(), ans[-200:])
    await hub.shutdown()
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
