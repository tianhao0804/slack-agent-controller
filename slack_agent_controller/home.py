"""The bot's Slack Home tab: one overview of everything that is going on.

Published with ``views.publish`` when you open the tab (``app_home_opened``)
and when you tap Refresh. Only allowlisted users see content.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import TYPE_CHECKING, Any

from .slackfmt import button, escape, inline_code

if TYPE_CHECKING:
    from .app import SlackFrontend

MAX_RECENT = 8
MAX_ITEMS = 10


def _section(text: str, accessory: dict[str, Any] | None = None) -> dict[str, Any]:
    b: dict[str, Any] = {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}
    if accessory:
        b["accessory"] = accessory
    return b


def _open(url: str | None, aid: str) -> dict[str, Any] | None:
    if not url:
        return None
    return {"type": "button", "action_id": aid, "url": url, "text": {"type": "plain_text", "text": "Open"}}


def _ctx(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text[:2900]}]}


def denied_view() -> dict[str, Any]:
    return {"type": "home", "blocks": [_section(":lock: This bot is limited to its allowlisted users.")]}


async def gpu_lines(front: SlackFrontend) -> list[str]:
    out = await front.hub._run("nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader,nounits")
    lines = []
    for raw in out.splitlines():
        f = [x.strip() for x in raw.split(",")]
        if len(f) == 4 and all(x.replace(".", "").isdigit() for x in f):
            idx, util, used, total = f
            bar = "█" * round(int(util) / 20) + "·" * (5 - round(int(util) / 20))
            lines.append(f"GPU{idx} {bar} {int(util):>3}%  {int(used) / 1024:4.1f}/{int(total) / 1024:.0f} GB")
    return lines or [out[:200] or "nvidia-smi: no output"]


async def build_home(front: SlackFrontend, user: str) -> dict[str, Any]:
    hub, cfg = front.hub, front.cfg
    blocks: list[dict[str, Any]] = []
    now = time.strftime("%H:%M")
    blocks.append({"type": "header", "text": {"type": "plain_text", "text": f"Agent Controller · {cfg.node_name}"[:150]}})
    blocks.append({"type": "actions", "block_id": "sac_home_top", "elements": [
        button(":arrows_counterclockwise: Refresh", "sac_home_refresh", "refresh"),
        button(":sparkles: New session (DM)", "sac_home_new", "default")]})
    blocks.append(_ctx(f"Updated {now} · default mode `{cfg.default_mode}` · model `{cfg.model or 'default'}`"))

    # -- needs you ---------------------------------------------------------
    needs: list[dict[str, Any]] = []
    for p in list(hub.approvals.values())[:MAX_ITEMS]:
        link = await front._permalink(p.channel, p.ts) if p.ts else None
        line = p.line or p.tool
        title = await _title(front, p.session.rec.session_id)
        needs.append(_section(f":lock: *Approval:* {escape(p.tool)} in _{escape(title)}_ · {inline_code(line[:80])}",
                              _open(link, f"sac_home_open_a{p.id[:6]}")))
    for q in list(hub.questions.values())[:MAX_ITEMS]:
        link = await front._permalink(q.channel, q.ts) if q.ts else None
        needs.append(_section(f":question: *Question:* {escape(q.text_of(0)[:120])}",
                              _open(link, f"sac_home_open_q{q.id[:6]}")))
    blocks.append({"type": "divider"})
    blocks.append(_section(f"*Needs you* ({len(hub.approvals) + len(hub.questions)})" if needs
                           else "*Needs you* — nothing right now :white_check_mark:"))
    blocks += needs

    # -- active sessions -------------------------------------------------------
    recs = sorted(hub.store.records.values(), key=lambda r: r.last_active, reverse=True)
    titles = await asyncio.to_thread(_titles_for, [r.session_id for r in recs[: MAX_ITEMS + MAX_RECENT]])

    def active(r: Any) -> bool:
        s = hub.sessions.get(r.key)
        locked = bool(hub.mirror and hub.mirror.locked(r.session_id))
        return locked or bool(s and (s.busy or s.bg_tasks or s.pending))

    act = [r for r in recs if active(r)][:MAX_ITEMS]
    blocks.append({"type": "divider"})
    blocks.append(_section(f"*Active* ({len(act)})" if act else "*Active* — no session is running"))
    for r in act:
        blocks.append(await _session_line(front, r, titles))

    # -- jobs ------------------------------------------------------------------
    if front.jobs is not None:
        ids = await asyncio.to_thread(front.jobs.recent, 6)
        live = [j for j in ids if front.jobs.cli.read_status(j).get("state") in ("queued", "running")]
        if ids:
            blocks.append({"type": "divider"})
            blocks.append(_section(f"*Jobs* ({len(live)} running or queued)"))
            for jid in ids:
                line = await asyncio.to_thread(front.jobs.describe, jid)
                blocks.append(_section(inline_code(line[:150]), button("Log", f"sac_job_log_h{jid}", jid)))

    # -- node ------------------------------------------------------------------
    blocks.append({"type": "divider"})
    blocks.append(_section("*GPUs*\n```" + escape("\n".join(await gpu_lines(front))) + "```"))

    # -- recent ------------------------------------------------------------------
    recent = [r for r in recs if r not in act][:MAX_RECENT]
    if recent:
        blocks.append({"type": "divider"})
        blocks.append(_section("*Recent sessions*"))
        for r in recent:
            blocks.append(await _session_line(front, r, titles))

    # -- projects ---------------------------------------------------------------
    if cfg.projects:
        blocks.append({"type": "divider"})
        blocks.append(_section("*Projects* — a new session opens in your DM with the bot"))
        for alias, d in cfg.projects[:MAX_ITEMS]:
            blocks.append(_section(f"*{escape(alias)}* · {inline_code(_short(d))}",
                                   button(":sparkles: New session", f"sac_home_proj_{alias}", alias)))

    foot = [f"{len(hub.store.records)} sessions stored"]
    if cfg.token_expires:
        days = (cfg.token_expires - dt.date.today()).days
        foot.append(f"API key ({cfg.provider}) expires {cfg.token_expires} ({days} days)")
    blocks.append(_ctx(" · ".join(foot)))
    return {"type": "home", "blocks": blocks[:100]}


async def _session_line(front: SlackFrontend, r: Any, titles: dict[str, str]) -> dict[str, Any]:
    where = front._project_of(r.cwd) or _short(r.cwd)
    title = escape((titles.get(r.session_id) or "(untitled)").splitlines()[0][:70])
    link = await front._permalink(r.channel, r.thread_ts)
    text = f"{front._session_state(r)} · *{title}*\n{inline_code(where)} · {r.turns} turns · `{r.session_id[:8]}`"
    return _section(text, _open(link, f"sac_home_open_s{r.session_id[:8]}"))


async def _title(front: SlackFrontend, sid: str) -> str:
    return (await asyncio.to_thread(_titles_for, [sid])).get(sid) or sid[:8]


def _titles_for(ids: list[str]) -> dict[str, str]:
    from .app import _titles

    return _titles(ids)


def _short(p: str) -> str:
    from .app import _short_path

    return _short_path(p)
