"""Настройки приложения из окружения и .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Рантайм-конфигурация деска."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    home: Path = Field(default=PROJECT_ROOT, alias="REFLEXDESK_HOME")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="REFLEXDESK_DATA_DIR")

    deepseek_api_key: str
    router_config: Path = Field(
        default=Path(__file__).resolve().parent / "router" / "config.yaml",
        alias="ROUTER_CONFIG",
    )

    database_url: str = ""
    checkpoint_url: str = ""
    qdrant_url: str = ""

    embedding_model: str = "BAAI/bge-m3"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    gateway_threshold: float = 0.55
    gateway_reasoning_gates: str = ""
    gateway_readout: Literal["one_forward", "l0"] = "one_forward"

    sec_user_agent: str
    fred_api_key: str = ""

    risk_officer_always: bool = False
    max_validation_rounds: int = 2
    collector_max_tool_calls: int = 12
    analyst_max_iterations: int = 3

    sandbox_socket: str = "/run/sandbox/sandbox.sock"

    tracing_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    @property
    def prompts_dir(self) -> Path:
        return self.home / "prompts"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'reflexdesk.db').as_posix()}"

    @property
    def resolved_checkpoint_url(self) -> str:
        return self.checkpoint_url or (self.data_dir / "checkpoints.db").as_posix()

    @property
    def resolved_qdrant_location(self) -> str:
        return self.qdrant_url or (self.data_dir / "qdrant").as_posix()

    @property
    def fred_enabled(self) -> bool:
        return bool(self.fred_api_key)

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key and self.langfuse_host)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон настроек процесса."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings()
    for path in (settings.data_dir, settings.reports_dir, settings.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
