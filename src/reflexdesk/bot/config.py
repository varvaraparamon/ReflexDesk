"""Настройки Telegram-бота: токен, белый список чатов, адрес API деска."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Конфигурация бота из окружения."""

    model_config = SettingsConfigDict(extra="ignore")

    telegram_bot_token: str
    telegram_allowed_chat_ids: str
    desk_api_url: str = "http://app:8000"
    state_path: Path = Field(default=Path("/state/bot_state.json"), alias="BOT_STATE_PATH")
    poll_runs_s: float = 15.0
    poll_predictions_s: float = 3600.0

    @property
    def allowed_chat_ids(self) -> frozenset[int]:
        return frozenset(int(x) for x in self.telegram_allowed_chat_ids.split(",") if x.strip())


@lru_cache(maxsize=1)
def get_bot_settings() -> BotSettings:
    """Синглтон настроек бота."""
    return BotSettings()
