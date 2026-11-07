"""LangChain-адаптер тиров роутера (для LangMem): та же конфигурация, учёт стоимости и фолбэки."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult as LCResult
from langchain_openai import ChatOpenAI

from reflexdesk.router.router import LLMRouter


class AccountingCallback(BaseCallbackHandler):
    """Пишет токены и стоимость LangChain-вызовов в метрики и ledger роутера."""

    def __init__(self, router: LLMRouter, tier: str) -> None:
        self._router = router
        self._tier = tier
        self._started: dict[UUID, float] = {}
        self.cost_usd = 0.0

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID, **_: Any) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LCResult, *, run_id: UUID, **_: Any) -> None:
        latency = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.cost_usd += self._router.account(
            self._tier,
            model=(response.llm_output or {}).get("model_name", self._router.tier(self._tier).model),
            tokens_in=usage.get("prompt_tokens", 0) or 0,
            cached_in=usage.get("prompt_cache_hit_tokens", 0) or 0,
            tokens_out=usage.get("completion_tokens", 0) or 0,
            latency_s=latency,
        )


def _single(router: LLMRouter, tier: str, callback: AccountingCallback) -> ChatOpenAI:
    cfg = router.tier(tier)
    provider = router.config.provider_of(tier)
    extras = router.config.request_extras(tier)
    return ChatOpenAI(
        model=tier if router.config.uses_proxy else cfg.model,
        base_url=provider.base_url,
        api_key=provider.api_key(),
        timeout=cfg.timeout_s,
        max_retries=router.config.retry.attempts,
        extra_body=extras.get("extra_body"),
        callbacks=[callback],
    )


def chat_model(router: LLMRouter, tier: str) -> tuple[ChatOpenAI, AccountingCallback]:
    """ChatOpenAI для тира и callback учёта его стоимости."""
    callback = AccountingCallback(router, tier)
    return _single(router, tier, callback), callback
