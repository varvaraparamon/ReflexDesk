"""Схема конфига роутера, подстановка env и проверка инвариантов."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

LOGICAL_TIERS = ("gateway", "tier0", "tier1", "tier2", "tier3")


class Price(BaseModel):
    """Цена в USD за 1M токенов."""

    input: float = 0.0
    cached_input: float = 0.0
    output: float = 0.0


class Provider(BaseModel):
    """OpenAI-совместимый эндпоинт."""

    base_url: str
    api_key_env: str | None = None
    structured: Literal["json_schema", "json_object"] = "json_object"
    reported_cost_currency: str | None = None

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else "not-needed"


class TierConfig(BaseModel):
    """Маппинг логического тира на физическую модель."""

    provider: str
    model: str
    family: str
    mode: Literal["logits", "instruct", "reasoning"]
    extra_body: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 30.0
    cache_ttl_s: float | None = None
    fallback: str | None = None
    fallback_flag: str | None = None
    budget_per_run_usd: float | None = None
    price_per_mtok: Price = Field(default_factory=Price)


class RetryPolicy(BaseModel):
    attempts: int = 2
    backoff_s: float = 1.0


class RouterConfig(BaseModel):
    """Полный конфиг роутера."""

    proxy: Provider | None = None
    providers: dict[str, Provider]
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    fx_per_usd: dict[str, float] = Field(default_factory=lambda: {"USD": 1.0})
    tiers: dict[str, TierConfig]

    @model_validator(mode="after")
    def _check_invariants(self) -> RouterConfig:
        missing = set(LOGICAL_TIERS) - set(self.tiers)
        if missing:
            raise ValueError(f"router config misses tiers: {sorted(missing)}")
        for name, provider in self.providers.items():
            currency = provider.reported_cost_currency
            if currency and currency not in self.fx_per_usd:
                raise ValueError(f"provider {name}: no fx rate for {currency}")
        for name, tier in self.tiers.items():
            if tier.provider not in self.providers:
                raise ValueError(f"tier {name}: unknown provider {tier.provider}")
            if tier.fallback and tier.fallback not in self.tiers:
                raise ValueError(f"tier {name}: unknown fallback {tier.fallback}")
        producer, verifier = self.tiers["tier2"], self.tiers["tier3"]
        if producer.model == verifier.model:
            raise ValueError("tier2 and tier3 must map to different models (cross-check rule)")
        if producer.family == verifier.family:
            raise ValueError("tier3 must belong to a different model family than tier2 (cross-check rule)")
        if verifier.mode != "reasoning" or producer.mode == verifier.mode:
            raise ValueError("tier3 must run in reasoning mode, tier2 in a different mode")
        self._check_fallback_cycles()
        return self

    def _check_fallback_cycles(self) -> None:
        for start in self.tiers:
            seen, cursor = set(), start
            while cursor:
                if cursor in seen:
                    raise ValueError(f"fallback cycle through {cursor}")
                seen.add(cursor)
                cursor = self.tiers[cursor].fallback

    @property
    def uses_proxy(self) -> bool:
        return bool(self.proxy and self.proxy.base_url)

    def request_extras(self, tier: str) -> dict[str, Any]:
        """Провайдерские параметры запроса (и напрямую, и через LiteLLM — единый источник)."""
        extra_body = self.tiers[tier].extra_body
        return {"extra_body": extra_body} if extra_body else {}

    def structured_mode(self, tier: str) -> str:
        """Режим structured output задаёт реальный провайдер тира, даже за прокси."""
        return self.providers[self.tiers[tier].provider].structured

    def provider_of(self, tier: str) -> Provider:
        """Провайдер, через который физически идёт вызов тира."""
        return self.proxy if self.uses_proxy else self.providers[self.tiers[tier].provider]


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_router_config(path: Path) -> RouterConfig:
    """Читает YAML, подставляет ${VAR:-default}, валидирует."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RouterConfig.model_validate(_interpolate(raw))
