"""Telegram-бот — основная точка взаимодействия с деском: запросы, статусы, отчёты, прогнозы."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, Message, TelegramObject

from reflexdesk.bot import messages
from reflexdesk.bot.client import DeskBusy, DeskClient
from reflexdesk.bot.config import BotSettings, get_bot_settings
from reflexdesk.bot.notifier import Notifier, requester_of
from reflexdesk.bot.state import BotState
from reflexdesk.obs.logging import get_logger

log = get_logger("bot")

DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
RUN_LOOKUP_LIMIT = 50
COMMANDS = [
    BotCommand(command="analyze", description="Запустить анализ: /analyze NVDA [ГГГГ-ММ-ДД]"),
    BotCommand(command="runs", description="Мои последние анализы"),
    BotCommand(command="status", description="Статус анализа по id"),
    BotCommand(command="report", description="Отчёт и PDF по id"),
    BotCommand(command="predictions", description="Мои прогнозы и их сверка"),
    BotCommand(command="metrics", description="Качество прогнозов деска"),
    BotCommand(command="help", description="Как пользоваться"),
]


class AllowlistMiddleware(BaseMiddleware):
    """Пропускает только чаты из TELEGRAM_ALLOWED_CHAT_IDS."""

    def __init__(self, allowed: frozenset[int]) -> None:
        self._allowed = allowed

    async def __call__(self, handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
                       event: TelegramObject, data: dict[str, Any]) -> Any:
        if isinstance(event, Message) and event.chat.id not in self._allowed:
            log.warning("access denied", chat_id=event.chat.id)
            await event.answer(messages.denied(event.chat.id))
            return None
        return await handler(event, data)


def split_query(text: str) -> tuple[str, date | None]:
    """Запрос и дата среза (первая ГГГГ-ММ-ДД в тексте)."""
    match = DATE_PATTERN.search(text)
    if not match:
        return text.strip(), None
    return (text[: match.start()] + text[match.end():]).strip(), date.fromisoformat(match.group(1))


def build_router(client: DeskClient, notifier: Notifier) -> Router:
    """Хендлеры команд и свободного текста."""
    router = Router()

    async def resolve(message: Message, ref: str | None) -> str | None:
        if not ref:
            await message.answer("Укажите id анализа, например: <code>/status 1a2b3c4d</code>. Список — /runs")
            return None
        runs = await client.runs(requester_of(message.chat.id), limit=RUN_LOOKUP_LIMIT)
        found = [r["run_id"] for r in runs if r["run_id"].startswith(ref.strip())]
        if not found:
            await message.answer("Анализ с таким id не найден среди ваших. Список — /runs")
            return None
        return found[0]

    async def start_analysis(message: Message, text: str) -> None:
        query, as_of = split_query(text)
        if not query:
            await message.answer("Напишите, что проанализировать, например: <code>Проанализируй AAPL</code>")
            return
        if as_of and as_of > date.today():
            await message.answer("Дата среза не может быть в будущем.")
            return
        try:
            run_id = await client.analyze(query, requester_of(message.chat.id), as_of)
        except DeskBusy:
            await message.answer("У вас уже есть незавершённый анализ — дождитесь отчёта. Статус: /runs")
            return
        await message.answer(messages.accepted(run_id, as_of.isoformat() if as_of else None))

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def help_(message: Message) -> None:
        await message.answer(messages.HELP)

    @router.message(Command("analyze"))
    async def analyze(message: Message, command: CommandObject) -> None:
        await start_analysis(message, command.args or "")

    @router.message(Command("runs"))
    async def runs(message: Message) -> None:
        await message.answer(messages.runs_list(await client.runs(requester_of(message.chat.id))))

    @router.message(Command("status"))
    async def status(message: Message, command: CommandObject) -> None:
        run_id = await resolve(message, command.args)
        if run_id:
            await message.answer(messages.run_status(await client.run(run_id)))

    @router.message(Command("report"))
    async def report(message: Message, command: CommandObject) -> None:
        run_id = await resolve(message, command.args)
        if run_id:
            await notifier.deliver_report(message.chat.id, run_id)

    @router.message(Command("predictions"))
    async def predictions(message: Message, command: CommandObject) -> None:
        ticker = (command.args or "").strip().upper() or None
        found = await client.predictions(requester=requester_of(message.chat.id), ticker=ticker)
        await message.answer(messages.predictions_list(found))

    @router.message(Command("metrics"))
    async def metrics(message: Message) -> None:
        await message.answer(messages.metrics(await client.metrics()))

    @router.message(F.text & ~F.text.startswith("/"))
    async def free_text(message: Message) -> None:
        await start_analysis(message, message.text)

    return router


async def run(settings: BotSettings) -> None:
    """Long polling + фоновые опросы готовых отчётов и сверенных прогнозов."""
    bot = Bot(settings.telegram_bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    client = DeskClient(settings.desk_api_url)
    notifier = Notifier(bot, client, BotState(settings.state_path), settings.allowed_chat_ids)
    dispatcher = Dispatcher()
    dispatcher.message.outer_middleware(AllowlistMiddleware(settings.allowed_chat_ids))
    dispatcher.include_router(build_router(client, notifier))
    await bot.set_my_commands(COMMANDS)
    background = [
        asyncio.create_task(notifier.run_forever(notifier.poll_runs, settings.poll_runs_s)),
        asyncio.create_task(notifier.run_forever(notifier.poll_predictions, settings.poll_predictions_s)),
    ]
    log.info("bot started", chats=len(settings.allowed_chat_ids))
    try:
        await dispatcher.start_polling(bot)
    finally:
        for task in background:
            task.cancel()
        await client.close()
        await bot.session.close()


def main() -> None:
    asyncio.run(run(get_bot_settings()))
