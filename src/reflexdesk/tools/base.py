"""Реестр тулов: регистрация, инструментирование, контекст среза данных as_of."""

from __future__ import annotations

import functools
import inspect
import time
import typing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, Field, create_model

from reflexdesk.obs import metrics
from reflexdesk.obs.logging import get_logger
from reflexdesk.obs.tracing import observe
from reflexdesk.schemas import Evidence

log = get_logger("tools")

_as_of: ContextVar[datetime | None] = ContextVar("tools_as_of", default=None)


class Document(BaseModel):
    """Текстовый материал для LLM-извлечения фактов."""

    title: str
    text: str
    url: str
    published_at: datetime | None = None


class ToolResult(BaseModel):
    """Унифицированный результат тула."""

    tool: str
    ok: bool = True
    data: Any = None
    evidence: list[Evidence] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0

    def brief(self, limit: int = 1500) -> str:
        """Короткое текстовое представление для контекста гейтов."""
        if not self.ok:
            return f"{self.tool}: ERROR {self.error}"
        head = f"{self.tool}: ok, facts={len(self.evidence)}, docs={len(self.documents)}"
        facts = "; ".join(f"{e.fact}={e.value}" for e in self.evidence[:10])
        return f"{head}. {facts}"[:limit]


class ToolError(RuntimeError):
    """Ожидаемая ошибка тула (неверные аргументы, нет данных)."""


@contextmanager
def data_cutoff(as_of: datetime | None) -> Iterator[None]:
    """Устанавливает срез данных для тулов на время блока."""
    token = _as_of.set(as_of)
    try:
        yield
    finally:
        _as_of.reset(token)


def as_of() -> datetime:
    """Текущий срез данных (по умолчанию — сейчас)."""
    return _as_of.get() or datetime.now(UTC)


def as_of_date() -> date:
    return as_of().date()


def is_historical(tolerance_days: int = 3) -> bool:
    """Прогон по исторической дате (look-ahead защита строже)."""
    return (datetime.now(UTC) - as_of()).days > tolerance_days


def clamp_date(value: date | str | None) -> date:
    """Дата не позже среза as_of."""
    cutoff = as_of_date()
    if value is None:
        return cutoff
    parsed = date.fromisoformat(value) if isinstance(value, str) else value
    return min(parsed, cutoff)


ToolFn = Callable[..., ToolResult]
REGISTRY: dict[str, ToolFn] = {}
DESCRIPTIONS: dict[str, str] = {}


def tool(name: str, description: str) -> Callable[[Callable[..., ToolResult]], ToolFn]:
    """Регистрирует тул: логирование, метрики, трейс-спан, перехват ошибок в ToolResult."""

    def decorator(fn: Callable[..., ToolResult]) -> ToolFn:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            started = time.perf_counter()
            with observe(f"tool:{name}", "tool", input={"args": args, "kwargs": kwargs}) as span:
                try:
                    result = fn(*args, **kwargs)
                except (ToolError, ValueError, KeyError) as exc:
                    result = ToolResult(tool=name, ok=False, error=f"{type(exc).__name__}: {exc}")
                except Exception as exc:
                    log.exception("tool crashed", tool=name)
                    result = ToolResult(tool=name, ok=False, error=f"{type(exc).__name__}: {exc}")
                result.latency_ms = int((time.perf_counter() - started) * 1000)
                span.update(output=result.brief(), metadata={"ok": result.ok, "latency_ms": result.latency_ms})
            status = "ok" if result.ok else "error"
            metrics.tool_calls_total.labels(name, status).inc()
            log.info("tool call", tool=name, status=status, latency_ms=result.latency_ms, error=result.error)
            return result

        REGISTRY[name] = wrapper
        DESCRIPTIONS[name] = description
        return wrapper

    return decorator


def get_tool(name: str) -> ToolFn:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool {name!r}")
    return REGISTRY[name]


def args_schema(name: str) -> type[BaseModel]:
    """Pydantic-схема аргументов тула из его сигнатуры (для structured-генерации аргументов)."""
    fn = get_tool(name)
    target = getattr(fn, "__wrapped__", fn)
    hints = typing.get_type_hints(target)
    fields = {
        param.name: (hints.get(param.name, str), ... if param.default is param.empty else param.default)
        for param in inspect.signature(target).parameters.values()
    }
    return create_model(f"{name}_args", __doc__=DESCRIPTIONS[name], **fields)
