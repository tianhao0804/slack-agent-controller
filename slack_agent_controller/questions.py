"""Claude's clarifying questions (AskUserQuestion) as a Slack card.

The CLI calls ``can_use_tool("AskUserQuestion", {"questions": [...]})``; the
answer goes back as ``updated_input = {"questions": ..., "answers": {question
text: label or [labels]}}`` (Agent SDK docs, "Handle clarifying questions").
Each call has 1-4 questions with 2-4 options. Single-choice questions get one
button per option, multi-choice ones checkboxes plus Submit; a plain reply in
the thread answers the first open question in your own words; Skip lets Claude
continue on its own judgement.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from .policy import redact_log
from .slackfmt import BLOCK_TEXT_LIMIT, button, escape

SKIP_RESPONSE = ("The user skipped these questions. Continue with your best judgement and say briefly what you "
                 "assumed.")


@dataclass
class PendingQuestion:
    id: str
    session_key: str
    owner: str
    questions: list[dict[str, Any]]
    future: asyncio.Future[tuple[str, Any]]
    channel: str = ""
    ts: str | None = None
    answers: dict[str, Any] = field(default_factory=dict)
    selections: dict[int, list[str]] = field(default_factory=dict)  # multi-choice, before Submit

    # Retain staged replies only while the question card is pending.
    attachment_labels: dict[int, str] = field(default_factory=dict)
    attached_replies: dict[str, tuple[str, str, str]] = field(default_factory=dict)

    # -- answering ---------------------------------------------------------
    def text_of(self, qi: int) -> str:
        return str(self.questions[qi].get("question") or f"Question {qi + 1}")

    def labels(self, qi: int) -> list[str]:
        opts = self.questions[qi].get("options") or []
        return [str(o.get("label") or "") for o in opts if isinstance(o, dict)]

    def open_index(self) -> int | None:
        return next((i for i in range(len(self.questions)) if self.text_of(i) not in self.answers), None)

    def answer(self, qi: int, value: Any, *, label: str = "") -> bool:
        """Record an answer; resolves the future once every question has one. True if accepted."""
        if self.future.done() or not 0 <= qi < len(self.questions) or self.text_of(qi) in self.answers:
            return False
        if label:
            self.attachment_labels[qi] = label
        self.answers[self.text_of(qi)] = value
        if self.open_index() is None:
            self.future.set_result(("answers", dict(self.answers)))
        return True

    def answer_free_text(self, text: str) -> bool:
        qi = self.open_index()
        return qi is not None and self.answer(qi, text.strip())

    # -- Slack card --------------------------------------------------------
    def blocks(self, final: str | None = None) -> list[dict[str, Any]]:
        n = len(self.questions)
        head = f"<@{self.owner}> :question: *Claude has {'a question' if n == 1 else f'{n} questions'}*"
        if final:
            head = f":question: *Claude asked {'a question' if n == 1 else f'{n} questions'}* — {final}"
        out: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": head}}]
        for qi, q in enumerate(self.questions):
            header = str(q.get("header") or "").strip()
            multi = bool(q.get("multiSelect"))
            title = (f"*{escape(header)}* — " if header else "") + escape(self.text_of(qi))
            if multi and self.text_of(qi) not in self.answers and not final:
                title += "  _(choose any, then Submit)_"
            out.append({"type": "section", "text": {"type": "mrkdwn", "text": title[:BLOCK_TEXT_LIMIT]}})
            opts = [o for o in q.get("options") or [] if isinstance(o, dict)]
            desc = "\n".join(f"• *{escape(str(o.get('label', '')))}* — {escape(str(o.get('description', '')))}"
                             for o in opts if o.get("description"))
            if desc and self.text_of(qi) not in self.answers and not final:
                out.append({"type": "context", "elements": [{"type": "mrkdwn", "text": desc[:BLOCK_TEXT_LIMIT]}]})
            if self.text_of(qi) in self.answers:
                ans = self.answers[self.text_of(qi)]
                shown = self.attachment_labels.get(qi) or (", ".join(ans) if isinstance(ans, list) else str(ans))
                out.append({"type": "context", "elements": [{"type": "mrkdwn",
                            "text": f":white_check_mark: {escape(redact_log(shown))[:BLOCK_TEXT_LIMIT - 30]}"}]})
                continue
            if final:
                continue
            if multi:
                checks = {"type": "checkboxes", "action_id": f"sac_qa_multi_{qi}", "options": [
                    {"text": {"type": "mrkdwn", "text": escape(lab)[:150]}, "value": f"{self.id}|{qi}|{oi}"}
                    for oi, lab in enumerate(self.labels(qi)[:10])]}
                out.append({"type": "actions", "block_id": f"sac_qa_{qi}", "elements": [
                    checks, button("Submit", f"sac_qa_submit_{qi}", f"{self.id}|{qi}", "primary")]})
            else:
                out.append({"type": "actions", "block_id": f"sac_qa_{qi}", "elements": [
                    button(lab or f"Option {oi + 1}", f"sac_qa_pick_{qi}_{oi}", f"{self.id}|{qi}|{oi}")
                    for oi, lab in enumerate(self.labels(qi)[:5])]})
        if not final:
            out.append({"type": "context", "elements": [{"type": "mrkdwn", "text":
                        "Or reply in this thread to answer the first open question in your own words."}]})
            out.append({"type": "actions", "block_id": "sac_qa_end",
                        "elements": [button("Skip", "sac_qa_skip", self.id)]})
        return out[:50]


def new_question(session_key: str, owner: str, questions: Any, loop: asyncio.AbstractEventLoop) -> PendingQuestion:
    qs = [q for q in (questions or []) if isinstance(q, dict)][:4]
    return PendingQuestion(id=uuid.uuid4().hex[:12], session_key=session_key, owner=owner, questions=qs,
                           future=loop.create_future())
