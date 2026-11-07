"""Telegram-бот: разбор запроса, доступ, доставка отчётов и сверенных прогнозов, повторная доставка не происходит."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

from aiogram.types import Chat, Message

from reflexdesk.bot import messages
from reflexdesk.bot.app import AllowlistMiddleware, split_query
from reflexdesk.bot.notifier import Notifier, requester_of
from reflexdesk.bot.state import BotState
from reflexdesk.schemas import utcnow

REPORT_MD = "# MSFT\n## Резюме\nДеск ожидает рост выручки 12% [1].\n## Прогнозы\n| MSFT | up | 90 | 0.6 | x |\n"


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.documents: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))

    async def send_document(self, chat_id: int, document) -> None:
        self.documents.append((chat_id, document.filename))


class FakeClient:
    def __init__(self, runs: list[dict], predictions: list[dict]) -> None:
        self._runs = runs
        self._predictions = predictions

    async def runs(self, requester: str, limit: int = 10) -> list[dict]:
        return self._runs

    async def run(self, run_id: str) -> dict:
        run = next(r for r in self._runs if r["run_id"] == run_id)
        return {"run_id": run_id, "status": run["status"], "ticker": "MSFT", "evidence_count": 19, "validation_round": 1,
                "validation": {"verdict": "ok"}, "flags": [], "cost_usd": 0.011, "errors": ["boom"], "as_of_date": None}

    async def report(self, run_id: str) -> dict:
        return {"markdown": REPORT_MD, "pdf_url": "x"}

    async def report_pdf(self, run_id: str) -> bytes:
        return b"%PDF"

    async def predictions(self, **filters) -> list[dict]:
        since = filters.get("verified_since")
        if since is None:
            return self._predictions
        return [p for p in self._predictions if p.get("verified_at") and p["verified_at"] > since.isoformat()]


def _prediction(**extra) -> dict:
    return {"ticker": "MSFT", "direction": "up", "horizon_days": 90, "confidence": 0.6, "thesis": "рост",
            "check_at": "2024-05-30T23:59:59+00:00", "status": "pending", "actual_return": None, "verified_at": None, **extra}


def test_split_query_extracts_as_of():
    assert split_query("Проанализируй MSFT 2024-03-01") == ("Проанализируй MSFT", date(2024, 3, 1))
    assert split_query("Проанализируй NVDA") == ("Проанализируй NVDA", None)


def test_report_card_strips_source_refs():
    summary = {"run_id": "abcdef123456", "status": "done", "ticker": "MSFT", "evidence_count": 19, "validation_round": 1,
               "validation": {"verdict": "ok"}, "flags": [], "cost_usd": 0.011}
    text = messages.report_summary(summary, REPORT_MD, [_prediction()])
    assert "Резюме" in text and "[1]" not in text and "<code>abcdef12</code>" in text and "📈" in text


def test_finished_run_delivered_once(tmp_path):
    later = (utcnow() + timedelta(minutes=1)).isoformat()
    runs = [{"run_id": "r1", "status": "done", "updated_at": later}, {"run_id": "r2", "status": "running", "updated_at": later}]
    bot = FakeBot()
    notifier = Notifier(bot, FakeClient(runs, [_prediction()]), BotState(tmp_path / "s.json"), frozenset({42}))
    asyncio.run(notifier.poll_runs())
    asyncio.run(notifier.poll_runs())
    assert len(bot.documents) == 1 and bot.documents[0] == (42, "MSFT_r1.pdf")
    assert sum("Резюме" in t for _, t in bot.sent) == 1


def test_failed_run_reports_reason(tmp_path):
    later = (utcnow() + timedelta(minutes=1)).isoformat()
    bot = FakeBot()
    notifier = Notifier(bot, FakeClient([{"run_id": "r3", "status": "failed", "updated_at": later}], []),
                        BotState(tmp_path / "s.json"), frozenset({42}))
    asyncio.run(notifier.poll_runs())
    assert not bot.documents and "не удался" in bot.sent[0][1]


def test_verified_prediction_notified_once(tmp_path):
    later = (utcnow() + timedelta(minutes=1)).isoformat()
    bot = FakeBot()
    client = FakeClient([], [_prediction(status="miss", actual_return=-0.05, verified_at=later)])
    notifier = Notifier(bot, client, BotState(tmp_path / "s.json"), frozenset({42}))
    asyncio.run(notifier.poll_predictions())
    asyncio.run(notifier.poll_predictions())
    assert len(bot.sent) == 1 and "не сбылся" in bot.sent[0][1] and "-5.0%" in bot.sent[0][1]


def test_state_does_not_replay_history(tmp_path):
    earlier = (utcnow() - timedelta(days=1)).isoformat()
    bot = FakeBot()
    notifier = Notifier(bot, FakeClient([{"run_id": "old", "status": "done", "updated_at": earlier}], []),
                        BotState(tmp_path / "s.json"), frozenset({42}))
    asyncio.run(notifier.poll_runs())
    assert not bot.sent


def test_allowlist_blocks_unknown_chat():
    answers: list[str] = []
    handled: list[bool] = []

    async def handler(event, data):
        handled.append(True)

    message = Message.model_construct(chat=Chat.model_construct(id=7, type="private"), text="hi")
    object.__setattr__(message, "answer", lambda text: _record(answers, text))
    asyncio.run(AllowlistMiddleware(frozenset({42}))(handler, message, {}))
    assert not handled and "<code>7</code>" in answers[0]


async def _record(store: list[str], text: str) -> None:
    store.append(text)


def test_requester_format():
    assert requester_of(123) == "tg:123"


def test_interrupted_notified_once_then_done(tmp_path):
    state = BotState(tmp_path / "s.json")
    bot = FakeBot()
    t1 = (utcnow() + timedelta(minutes=1)).isoformat()
    t2 = (utcnow() + timedelta(minutes=2)).isoformat()
    t3 = (utcnow() + timedelta(minutes=3)).isoformat()
    for status, ts in (("interrupted", t1), ("interrupted", t2), ("done", t3)):
        client = FakeClient([{"run_id": "r9", "status": status, "updated_at": ts}], [])
        asyncio.run(Notifier(bot, client, state, frozenset({42})).poll_runs())
    assert sum("прерван" in t for _, t in bot.sent) == 1
    assert len(bot.documents) == 1
