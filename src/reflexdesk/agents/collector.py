"""Data Collector: цикл G2 (выбор тула) -> аргументы T1 -> вызов -> G3 при ошибке -> Evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from reflexdesk.agents.base import Agent, Usage, to_json
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import COLLECTOR
from reflexdesk.router.results import StructuredOutputError
from reflexdesk.schemas import Evidence, EvidenceBatch, Plan
from reflexdesk.state import DeskState
from reflexdesk.tools import (
    DESCRIPTIONS,
    ToolResult,
    args_schema,
    collector_tools,
    get_tool,
)
from reflexdesk.tools.base import Document, as_of

FINISH = "finish"
FINISH_DESCRIPTION = "every collection goal is covered by the tools already called"
ESCALATION_TIER = "tier2"
MAX_DOCS_FOR_EXTRACTION = 8
MEMORY_HITS = 5
MEMORY_PREFIX = "[память]"
MAX_TOOL_FAILURES = 2
DEFAULT_WINDOW_DAYS = 365


@dataclass
class Step:
    """Шаг сбора: тул, аргументы, краткий результат."""

    tool: str
    args: dict[str, Any]
    brief: str
    ok: bool


@dataclass
class Trajectory:
    """История шагов сборщика."""

    steps: list[Step] = field(default_factory=list)

    def recent(self, n: int = 2) -> str:
        if not self.steps:
            return "Шагов ещё не было."
        return "\n".join(f"{s.tool}({json.dumps(s.args, ensure_ascii=False)}) → {s.brief}" for s in self.steps[-n:])

    def succeeded(self) -> set[str]:
        return {s.tool for s in self.steps if s.ok}

    def exhausted(self, limit: int) -> set[str]:
        """Тулы, не давшие результата limit раз подряд (включая ретрай через G3)."""
        failures: dict[str, int] = {}
        for s in self.steps:
            failures[s.tool] = 0 if s.ok else failures.get(s.tool, 0) + 1
        return {tool for tool, count in failures.items() if count >= limit}


def tool_menu(tools: tuple[str, ...], unavailable: set[str], has_data: bool) -> dict[str, str]:
    """Варианты для G2: тулы без успешного вызова и не исчерпавшие попытки; finish — только когда есть данные."""
    menu = {name: DESCRIPTIONS[name] for name in tools if name not in unavailable}
    if not menu:
        return {}
    return menu | {FINISH: FINISH_DESCRIPTION} if has_data else menu


def tool_gate_context(tickers: list[str], goals: list[str], succeeded: list[str], tools: tuple[str, ...],
                      facts: int, recent: str) -> str:
    """Состояние для G2 — единый формат для прогона и для контрольного набора."""
    pending = [t for t in tools if t not in succeeded]
    goal_lines = "\n".join(f"- {g}" for g in goals)
    return (
        f"Tickers: {', '.join(tickers)}.\nCollection goals:\n{goal_lines}\n"
        f"Tools already called successfully: {', '.join(succeeded) or 'none'}.\n"
        f"Tools not called yet: {', '.join(pending) or 'none'}. Facts collected: {facts}.\n"
        f"Last steps:\n{recent}"
    )


class CollectorNode:
    """Сбор фактов в пределах лимита tool calls."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._agent: Agent = services.agent(COLLECTOR)
        self._max_calls = services.settings.collector_max_tool_calls

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        plan: Plan = state["plan"]
        tools = collector_tools()
        trajectory = Trajectory()
        evidence: list[Evidence] = []
        documents: list[Document] = []
        sector = state.get("sector")
        gate_log: list[dict] = []

        for _ in range(self._max_calls):
            menu = self._menu(tools, trajectory, has_data=bool(evidence or documents))
            if not menu:
                break
            decision = self._services.gates.choose_tool(self._gate_context(plan, trajectory, len(evidence), tools), menu)
            gate_log.append(decision.model_dump())
            if decision.pick == FINISH:
                break
            result, args = self._run_tool(decision.pick, plan, trajectory, usage, gate_log)
            trajectory.steps.append(Step(decision.pick, args, result.brief(400), result.ok))
            if not result.ok:
                usage.error(f"collector.{decision.pick}", result.error)
                continue
            evidence.extend(result.evidence)
            documents.extend(result.documents)
            if isinstance(result.data, dict):
                sector = sector or result.data.get("profile", {}).get("sector")

        self._remember(documents, plan.ticker)
        documents.extend(self._memory_documents(plan.ticker))
        evidence.extend(self._extract(documents, plan, usage))
        return {"evidence": _dedupe(evidence), "sector": sector, "gate_decisions": gate_log, **usage.as_update()}

    @staticmethod
    def _menu(tools: tuple[str, ...], trajectory: Trajectory, has_data: bool) -> dict[str, str]:
        return tool_menu(tools, trajectory.succeeded() | trajectory.exhausted(MAX_TOOL_FAILURES), has_data)

    @staticmethod
    def _gate_context(plan: Plan, trajectory: Trajectory, facts: int, tools: tuple[str, ...]) -> str:
        goals = [t.goal for t in plan.subtasks if t.assignee == "collector"]
        return tool_gate_context(plan.tickers, goals, sorted(trajectory.succeeded()), tools, facts, trajectory.recent())

    def _run_tool(self, tool: str, plan: Plan, trajectory: Trajectory, usage: Usage, gate_log: list[dict]) -> tuple[ToolResult, dict]:
        args = self._make_args(tool, plan, trajectory, usage, tier=COLLECTOR.tier)
        if args is None:
            return ToolResult(tool=tool, ok=False, error="arguments were not generated"), {}
        result = get_tool(tool)(**args)
        if result.ok:
            return result, args
        retry = self._services.gates.should_retry(result.error or "", f"{tool}({args})")
        gate_log.append(retry.model_dump())
        tier = COLLECTOR.tier if retry.pick == "yes" else ESCALATION_TIER
        fixed = self._make_args(tool, plan, trajectory, usage, tier=tier, error=f"{tool}({args}) → {result.error}")
        if fixed is None or fixed == args:
            return result, args
        return get_tool(tool)(**fixed), fixed

    def _make_args(self, tool: str, plan: Plan, trajectory: Trajectory, usage: Usage, *, tier: str,
                   error: str | None = None) -> dict | None:
        """Аргументы тула от LLM по схеме сигнатуры; при невалидном выводе — ошибка в errors и None."""
        cutoff = as_of().date()
        window_start = cutoff - timedelta(days=DEFAULT_WINDOW_DAYS)
        prompt = (
            f"Сформируй аргументы вызова инструмента {tool} ({DESCRIPTIONS[tool]}).\n"
            f"Тикеры: {', '.join(plan.tickers)}. Дата среза: {cutoff.isoformat()}.\n"
            f"Периоды: start строго раньше end, end не позже даты среза; "
            f"типичное окно — {window_start.isoformat()} … {cutoff.isoformat()}.\n"
            f"Задачи: {to_json([t.goal for t in plan.subtasks])}\nПоследние шаги:\n{trajectory.recent()}"
        )
        if error:
            prompt += f"\nПредыдущий вызов завершился ошибкой, исправь аргументы: {error}"
        try:
            result = self._services.router.structured(tier, self._agent.messages(prompt), args_schema(tool), metadata={"agent": "collector.args"})
        except StructuredOutputError as exc:
            usage.add_calls(exc.calls)
            usage.first_try_valid.append(False)
            usage.error(f"collector.args.{tool}", exc)
            return None
        usage.add_calls(result.calls)
        usage.first_try_valid.append(result.first_try_valid)
        return result.value.model_dump(exclude_none=True)

    def _extract(self, documents: list[Document], plan: Plan, usage: Usage) -> list[Evidence]:
        """LLM-извлечение фактов из текстов; факты с URL вне документов отбрасываются (нет источника — нет факта)."""
        if not documents:
            return []
        docs = documents[:MAX_DOCS_FOR_EXTRACTION]
        allowed = {d.url: d for d in docs}
        rendered = "\n\n".join(
            f"### {d.title}\nURL: {d.url}\nДата: {d.published_at.date() if d.published_at else 'н/д'}\n{d.text[:3000]}"
            for d in docs
        )
        prompt = (
            f"Извлеки факты, существенные для анализа {', '.join(plan.tickers)}, из документов ниже. "
            f"source_url — строго URL документа, из которого взят факт.\n\n{rendered}"
        )
        batch = self._agent.ask(prompt, EvidenceBatch, usage)
        if batch is None:
            return []
        kept = []
        for item in batch.evidence:
            if item.source_url not in allowed:
                continue
            item.published_at = item.published_at or allowed[item.source_url].published_at
            item.tool = item.tool or "extraction"
            kept.append(item)
        return kept

    def _memory_documents(self, ticker: str | None) -> list[Document]:
        """Ранее собранные документы тикера из L3 (только опубликованные до среза)."""
        if not ticker:
            return []
        hits = self._services.knowledge.documents.search(
            f"{ticker} выручка маржа риски гайденс", as_of=as_of(), ticker=ticker, top_k=MEMORY_HITS,
        )
        return [
            Document(
                title=f"{MEMORY_PREFIX} {h.payload.get('title', '')}", text=h.text, url=h.payload["url"],
                published_at=datetime.fromisoformat(h.payload["published_at"]),
            )
            for h in hits
        ]

    def _remember(self, documents: list[Document], ticker: str | None) -> None:
        """Запись собранных документов в L3 documents."""
        for doc in documents:
            self._services.knowledge.documents.ingest(
                doc.text, url=doc.url, title=doc.title, ticker=ticker,
                published_at=doc.published_at, kind="filing" if "sec.gov" in doc.url else "news",
            )


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for item in evidence:
        key = (item.fact, str(item.value), item.source_url)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
