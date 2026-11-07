"""LLM-роутер: логические тиры, ретраи, фолбэки, кэш, учёт стоимости."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, TypeVar

import openai
from pydantic import BaseModel

from reflexdesk.cache import TTLCache, get_cache, make_key
from reflexdesk.config import get_settings
from reflexdesk.obs import metrics
from reflexdesk.obs.context import current_run_id
from reflexdesk.obs.logging import get_logger
from reflexdesk.obs.tracing import observe
from reflexdesk.router.accounting import SpendLedger, compute_cost
from reflexdesk.router.config import RouterConfig, TierConfig, load_router_config
from reflexdesk.router.results import (
    BudgetExceeded,
    LLMResult,
    RouterError,
    StructuredOutputError,
    StructuredResult,
    TierUnavailable,
)
from reflexdesk.router.structured import (
    parse_structured,
    response_format,
    schema_instruction,
)

T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]

log = get_logger("router")

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_RETRYABLE_STATUS = frozenset({408, 409})


def is_transient(exc: openai.APIError) -> bool:
    """Временный сбой провайдера или прокси (в т.ч. 408 — таймаут LiteLLM)."""
    return isinstance(exc, _RETRYABLE) or getattr(exc, "status_code", None) in _RETRYABLE_STATUS


class LLMRouter:
    """Единая точка входа для всех LLM-вызовов деска."""

    def __init__(self, config: RouterConfig, cache: TTLCache | None = None) -> None:
        self.config = config
        self.ledger = SpendLedger()
        self._cache = cache
        self._clients: dict[str, openai.OpenAI] = {}

    def tier(self, name: str) -> TierConfig:
        return self.config.tiers[name]

    def fallback_chain(self, tier: str) -> list[str]:
        """Тир и его фолбэки по порядку."""
        chain, cursor = [], tier
        while cursor:
            chain.append(cursor)
            cursor = self.tier(cursor).fallback
        return chain

    def chat(
        self,
        tier: str,
        messages: Messages,
        *,
        max_tokens: int | None = None,
        schema: type[BaseModel] | None = None,
        logprobs: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Вызов тира; при сбоях транспорта или бюджета идёт по цепочке фолбэков (response_format — под каждый тир)."""
        flags: list[str] = []
        cursor: str | None = tier
        last_error: Exception | None = None
        while cursor:
            cfg = self.tier(cursor)
            try:
                self._check_budget(cursor, cfg)
                fmt = response_format(schema, self.config.structured_mode(cursor)) if schema else None
                result = self._call_with_retry(cursor, cfg, messages, max_tokens, fmt, logprobs, metadata)
                result.requested_tier = tier
                result.flags = flags
                return result
            except (TierUnavailable, BudgetExceeded) as exc:
                last_error = exc
                log.warning("tier failed, falling back", tier=cursor, fallback=cfg.fallback, error=str(exc))
                if cfg.fallback_flag:
                    flags.append(cfg.fallback_flag)
                if cursor != tier:
                    flags.append(f"fallback_from_{cursor}")
                cursor = cfg.fallback
                if cursor:
                    flags.append(f"served_by_{cursor}")
        raise TierUnavailable(f"tier {tier} and its fallbacks failed: {last_error}")

    def structured(
        self,
        tier: str,
        messages: Messages,
        schema: type[T],
        *,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredResult[T]:
        """Structured-вызов тира; один авто-ре-трай с ошибкой схемы."""
        convo = _with_schema(messages, schema)
        calls: list[LLMResult] = []
        error = ""
        for attempt in range(2):
            result = self.chat(tier, convo, max_tokens=max_tokens, schema=schema, metadata=metadata)
            calls.append(result)
            try:
                value = parse_structured(result.content, schema)
                return StructuredResult[schema](value=value, calls=calls, first_try_valid=attempt == 0)
            except ValueError as exc:
                error = str(exc)
                log.warning("structured output invalid", tier=tier, schema=schema.__name__, error=error)
                convo = [
                    *convo,
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"Ответ не прошёл валидацию схемы: {error}. Верни исправленный JSON."},
                ]
        raise StructuredOutputError(f"{schema.__name__}: {error}", calls)

    def _check_budget(self, tier: str, cfg: TierConfig) -> None:
        if cfg.budget_per_run_usd is None:
            return
        run_id = current_run_id.get()
        if self.ledger.spent(run_id, tier) >= cfg.budget_per_run_usd:
            raise BudgetExceeded(f"{tier} budget {cfg.budget_per_run_usd}$ exhausted for run {run_id}")

    def _call_with_retry(self, tier, cfg, messages, max_tokens, fmt, logprobs, metadata) -> LLMResult:
        cache_key = None
        if cfg.cache_ttl_s and self._cache is not None and logprobs is None:
            cache_key = make_key(cfg.model, messages, fmt, max_tokens)
            hit = self._cache.get(cache_key)
            if hit is not None:
                return LLMResult.model_validate({**hit, "cached": True, "cost_usd": 0.0, "latency_ms": 0})

        attempts = self.config.retry.attempts + 1
        for attempt in range(attempts):
            try:
                result = self._invoke(tier, cfg, messages, max_tokens, fmt, logprobs, metadata)
                if cache_key:
                    self._cache.set(cache_key, result.model_dump(), cfg.cache_ttl_s)
                return result
            except openai.APIError as exc:
                if not is_transient(exc):
                    raise RouterError(f"{tier}: {type(exc).__name__}: {exc}") from exc
                if attempt == attempts - 1:
                    raise TierUnavailable(f"{tier}: {type(exc).__name__}: {exc}") from exc
                log.warning("tier call failed, retrying", tier=tier, attempt=attempt + 1, error=f"{type(exc).__name__}: {exc}")
                time.sleep(self.config.retry.backoff_s * 2**attempt)
        raise TierUnavailable(tier)

    def _invoke(self, tier, cfg: TierConfig, messages, max_tokens, fmt, logprobs, metadata) -> LLMResult:
        params: dict[str, Any] = {
            "model": tier if self.config.uses_proxy else cfg.model,
            "messages": messages,
            "timeout": cfg.timeout_s,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        if fmt:
            params["response_format"] = fmt
        if logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = logprobs
        params.update(self.config.request_extras(tier))

        with observe(f"llm:{tier}", "generation", input=messages, model=cfg.model, metadata=metadata) as span:
            started = time.perf_counter()
            response = self._client(tier).chat.completions.create(**params)
            latency = time.perf_counter() - started
            result = self._to_result(tier, cfg, response, latency)
            span.update(
                output=result.content,
                usage_details={"input": result.tokens_in, "output": result.tokens_out},
                cost_details={"total": result.cost_usd},
                metadata={**(metadata or {}), "tier": tier, "latency_ms": result.latency_ms},
            )
        self._record(tier, cfg, result, latency)
        return result

    def _to_result(self, tier: str, cfg: TierConfig, response, latency: float) -> LLMResult:
        choice = response.choices[0]
        usage = response.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        cached_in = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        reasoning = getattr(choice.message, "reasoning_content", None) or (choice.message.model_extra or {}).get("reasoning")
        logprobs = None
        if choice.logprobs and choice.logprobs.content:
            logprobs = [item.model_dump() for item in choice.logprobs.content]
        return LLMResult(
            content=choice.message.content or "",
            reasoning=reasoning,
            requested_tier=tier,
            served_tier=tier,
            model=response.model or cfg.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost(cfg, usage, tokens_in, cached_in, tokens_out),
            latency_ms=int(latency * 1000),
            logprobs=logprobs,
        )

    def _cost(self, cfg: TierConfig, usage, tokens_in: int, cached_in: int, tokens_out: int) -> float:
        """Стоимость из ответа провайдера (в его валюте → USD), если он её сообщает; иначе по таблице цен тира."""
        currency = self.config.providers[cfg.provider].reported_cost_currency
        reported = (getattr(usage, "model_extra", None) or {}).get("cost") if usage is not None else None
        if currency and reported is not None:
            return float(reported) / self.config.fx_per_usd[currency]
        return compute_cost(cfg.price_per_mtok, tokens_in, cached_in, tokens_out)

    def account(self, tier: str, *, model: str, tokens_in: int, cached_in: int, tokens_out: int, latency_s: float) -> float:
        """Учёт вызова, сделанного в обход chat() (например, LangChain-клиентом); возвращает стоимость."""
        cfg = self.tier(tier)
        result = LLMResult(
            content="", requested_tier=tier, served_tier=tier, model=model, tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=compute_cost(cfg.price_per_mtok, tokens_in, cached_in, tokens_out), latency_ms=int(latency_s * 1000),
        )
        self._record(tier, cfg, result, latency_s)
        return result.cost_usd

    def _record(self, tier: str, cfg: TierConfig, result: LLMResult, latency: float) -> None:
        self.ledger.add(current_run_id.get(), tier, result.cost_usd)
        metrics.llm_latency_seconds.labels(tier).observe(latency)
        metrics.llm_cost_usd_total.labels(tier).inc(result.cost_usd)
        metrics.llm_tokens_total.labels(tier, "in").inc(result.tokens_in)
        metrics.llm_tokens_total.labels(tier, "out").inc(result.tokens_out)
        log.info(
            "llm call", tier=tier, model=result.model, tokens_in=result.tokens_in,
            tokens_out=result.tokens_out, cost_usd=round(result.cost_usd, 6), latency_ms=result.latency_ms,
        )

    def _client(self, tier: str) -> openai.OpenAI:
        provider = self.config.provider_of(tier)
        key = provider.base_url
        if key not in self._clients:
            self._clients[key] = openai.OpenAI(base_url=provider.base_url, api_key=provider.api_key(), max_retries=0)
        return self._clients[key]


def _with_schema(messages: Messages, schema: type[BaseModel]) -> Messages:
    instruction = schema_instruction(schema)
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": f"{messages[0]['content']}\n\n{instruction}"}, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    """Синглтон роутера из YAML-конфига."""
    settings = get_settings()
    return LLMRouter(load_router_config(settings.router_config), cache=get_cache("llm"))
