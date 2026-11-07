"""Базовые абстракции агентов: спецификация, учёт расходов узла, вызовы LLM по тиру."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TypeVar

from pydantic import BaseModel

from reflexdesk.prompting import PromptLibrary, SystemPrompt
from reflexdesk.router.results import LLMResult, StructuredOutputError, StructuredResult
from reflexdesk.router.router import LLMRouter
from reflexdesk.schemas import Evidence

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class AgentSpec:
    """Контракт агента: тир, скиллы, тулы, доп. документы промпта."""

    name: str
    tier: str
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()


@dataclass
class Usage:
    """Накопитель стоимости, флагов и ошибок узла для обновления DeskState."""

    cost_usd: float = 0.0
    flags: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    first_try_valid: list[bool] = field(default_factory=list)

    def add_calls(self, calls: list[LLMResult]) -> None:
        self.cost_usd += sum(c.cost_usd for c in calls)
        self.flags.extend(f for c in calls for f in c.flags)

    def error(self, where: str, exc: Exception | str) -> None:
        self.errors.append(f"{where}: {exc}")

    def as_update(self) -> dict[str, Any]:
        return {
            "cost_usd": self.cost_usd,
            "flags": list(dict.fromkeys(self.flags)),
            "errors": self.errors,
            "json_first_try": self.first_try_valid,
        }


class Agent:
    """Агент: системный промпт из библиотеки + вызовы своего тира через роутер."""

    def __init__(self, spec: AgentSpec, router: LLMRouter, library: PromptLibrary) -> None:
        self.spec = spec
        self._router = router
        self._library = library

    @cached_property
    def prompt(self) -> SystemPrompt:
        return self._library.build(self.spec.name, self.spec.skills, self.spec.tools, self.spec.documents)

    def messages(self, user: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.prompt.text}, *(history or []), {"role": "user", "content": user}]

    def ask(self, user: str, schema: type[T], usage: Usage, *, history: list[dict[str, str]] | None = None,
            max_tokens: int | None = None) -> T | None:
        """Structured-вызов; при невалидном выводе после ре-трая — ошибка в usage и None."""
        try:
            result: StructuredResult[T] = self._router.structured(
                self.spec.tier, self.messages(user, history), schema,
                max_tokens=max_tokens, metadata=self._metadata(),
            )
        except StructuredOutputError as exc:
            usage.add_calls(exc.calls)
            usage.first_try_valid.append(False)
            usage.error(self.spec.name, exc)
            return None
        usage.add_calls(result.calls)
        usage.first_try_valid.append(result.first_try_valid)
        return result.value

    def write(self, user: str, usage: Usage, *, max_tokens: int | None = None) -> str:
        """Свободный текстовый вызов (markdown)."""
        result = self._router.chat(self.spec.tier, self.messages(user), max_tokens=max_tokens, metadata=self._metadata())
        usage.add_calls([result])
        return result.content

    def _metadata(self) -> dict[str, Any]:
        return {"agent": self.spec.name, **self.prompt.trace_metadata()}


def to_json(value: Any) -> str:
    """Компактная JSON-сериализация артефактов для промптов."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, list):
        value = [v.model_dump(mode="json", exclude_none=True) if isinstance(v, BaseModel) else v for v in value]
    return json.dumps(value, ensure_ascii=False, default=str)


class EvidenceView:
    """Единая нумерация источников [n] для всех агентов прогона."""

    def __init__(self, evidence: list[Evidence], limit: int = 150) -> None:
        self.evidence = evidence[:limit]
        self.sources: list[str] = list(dict.fromkeys(e.source_url for e in self.evidence))
        self._index = {url: i + 1 for i, url in enumerate(self.sources)}

    def facts(self) -> str:
        lines = []
        for item in self.evidence:
            date = f", {item.published_at.date()}" if item.published_at else ""
            lines.append(f"- {item.fact}: {item.value} [{self._index[item.source_url]}{date}]")
        return "\n".join(lines) or "Фактов нет."

    def source_list(self) -> str:
        return "\n".join(f"{i}. {url}" for i, url in enumerate(self.sources, 1)) or "Источников нет."

    def render(self) -> str:
        return f"Факты (Evidence):\n{self.facts()}\n\nИсточники:\n{self.source_list()}"
