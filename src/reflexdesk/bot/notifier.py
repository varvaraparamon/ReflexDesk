"""Доставка результатов в Telegram: готовые отчёты и сверенные прогнозы."""

from __future__ import annotations

import asyncio

from aiogram import Bot
from aiogram.types import BufferedInputFile

from reflexdesk.bot import messages
from reflexdesk.bot.client import DeskClient
from reflexdesk.bot.state import RUNS_KEY, VERIFIED_KEY, BotState
from reflexdesk.obs.logging import get_logger
from reflexdesk.schemas import ensure_utc

log = get_logger("bot.notifier")

FINAL_STATUSES = frozenset({"done", "failed", "rejected", "interrupted"})


def requester_of(chat_id: int) -> str:
    """Идентификатор пользователя деска для чата Telegram."""
    return f"tg:{chat_id}"


class Notifier:
    """Опрос API деска и отправка уведомлений в чаты из белого списка."""

    def __init__(self, bot: Bot, client: DeskClient, state: BotState, chats: frozenset[int]) -> None:
        self._bot = bot
        self._client = client
        self._state = state
        self._chats = chats

    async def deliver_report(self, chat_id: int, run_id: str) -> None:
        """Карточка отчёта и PDF; для неуспешных прогонов — сообщение о причине."""
        summary = await self._client.run(run_id)
        if summary["status"] != "done":
            await self._bot.send_message(chat_id, messages.run_finished(summary))
            return
        report = await self._client.report(run_id)
        predictions = await self._client.predictions(run_id=run_id)
        await self._bot.send_message(chat_id, messages.report_summary(summary, report["markdown"], predictions))
        pdf = await self._client.report_pdf(run_id)
        name = f"{summary.get('ticker') or 'report'}_{messages.short(run_id)}.pdf"
        await self._bot.send_document(chat_id, BufferedInputFile(pdf, filename=name))

    async def poll_runs(self) -> None:
        """Прогоны, завершившиеся после последней отметки чата."""
        for chat_id in self._chats:
            since = self._state.mark(chat_id, RUNS_KEY)
            runs = await self._client.runs(requester_of(chat_id))
            finished = [r for r in runs if r["status"] in FINAL_STATUSES and ensure_utc(messages.iso(r["updated_at"])) > since]
            for run in sorted(finished, key=lambda r: r["updated_at"]):
                if self._state.notified_status(chat_id, run["run_id"]) != run["status"]:
                    await self.deliver_report(chat_id, run["run_id"])
                    self._state.remember_status(chat_id, run["run_id"], run["status"])
                self._state.advance(chat_id, RUNS_KEY, ensure_utc(messages.iso(run["updated_at"])))

    async def poll_predictions(self) -> None:
        """Прогнозы пользователя, сверенные Reflector после последней отметки."""
        for chat_id in self._chats:
            since = self._state.mark(chat_id, VERIFIED_KEY)
            fresh = await self._client.predictions(requester=requester_of(chat_id), verified_since=since)
            for prediction in sorted(fresh, key=lambda p: p["verified_at"]):
                await self._bot.send_message(chat_id, messages.verified(prediction))
                self._state.advance(chat_id, VERIFIED_KEY, ensure_utc(messages.iso(prediction["verified_at"])))

    async def run_forever(self, poll, interval_s: float) -> None:
        """Периодический опрос; ошибка одного цикла логируется, цикл продолжается."""
        while True:
            try:
                await poll()
            except Exception:
                log.exception("notifier cycle failed", poll=poll.__name__)
            await asyncio.sleep(interval_s)
