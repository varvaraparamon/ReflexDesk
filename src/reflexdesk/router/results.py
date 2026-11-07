"""Типы результатов и ошибки роутера."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel):
    """Результат LLM-вызова после ретраев и фолбэков."""

    content: str
    reasoning: str | None = None
    requested_tier: str
    served_tier: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    flags: list[str] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = None


class StructuredResult(BaseModel, Generic[T]):
    """Провалидированный structured output с учётом вызовов."""

    value: T
    calls: list[LLMResult]
    first_try_valid: bool

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def flags(self) -> list[str]:
        return sorted({f for c in self.calls for f in c.flags})


class RouterError(RuntimeError):
    """Базовая ошибка роутера."""


class TierUnavailable(RouterError):
    """Исчерпаны ретраи и фолбэки тира."""


class BudgetExceeded(RouterError):
    """Исчерпан бюджет тира на прогон."""


class StructuredOutputError(RouterError):
    """Ответ не прошёл валидацию схемы после ре-трая."""

    def __init__(self, message: str, calls: list[LLMResult]) -> None:
        super().__init__(message)
        self.calls = calls
