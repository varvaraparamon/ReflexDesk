"""Analyst: уроки L3 + Evidence -> итерации python_exec -> AnalysisDraft."""

from __future__ import annotations

from typing import Any

from reflexdesk.agents.base import EvidenceView, Usage, to_json
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import ANALYST
from reflexdesk.schemas import AnalysisDraft, AnalystStep, as_of_from_iso
from reflexdesk.state import DeskState
from reflexdesk.tools import get_tool

MAX_OUTPUT_CHARS = 3000


class AnalystNode:
    """Аналитический драфт с вычислениями в sandbox."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._agent = services.agent(ANALYST)
        self._max_iterations = services.settings.analyst_max_iterations

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        lessons, lesson_ids = self._lessons(state)
        view = EvidenceView(state.get("evidence", []))
        prompt = (
            f"Запрос: {state['query']}\nТикер: {state['ticker']}; сектор: {state.get('sector') or 'н/д'}; "
            f"дата среза: {state.get('as_of_date') or 'сегодня'}.\n"
            f"Задачи анализа: {to_json([t.goal for t in state['plan'].subtasks if t.assignee == 'analyst'])}\n\n"
            f"{view.render()}\n\nУроки прошлых прогонов:\n{lessons}"
        )
        history: list[dict[str, str]] = []
        computations: list[str] = []
        draft: AnalysisDraft | None = None
        user = prompt
        for iteration in range(self._max_iterations):
            if iteration == self._max_iterations - 1:
                user += "\n\nЭто последняя итерация: верни draft."
            step = self._agent.ask(user, AnalystStep, usage, history=history)
            if step is None:
                break
            if step.draft is not None:
                draft = step.draft
                break
            output = self._execute(step.code or "", usage)
            computations.append(output)
            history += [
                {"role": "user", "content": user},
                {"role": "assistant", "content": step.model_dump_json(exclude_none=True)},
            ]
            user = f"Результат python_exec:\n{output}"
        if draft is None:
            usage.error("analyst", "драфт не получен")
            return {"analysis_draft": None, "status": "failed", "lessons_used": lesson_ids, **usage.as_update()}
        draft.ticker = draft.ticker.upper() or state["ticker"]
        draft.computations = computations
        return {"analysis_draft": draft, "lessons_used": lesson_ids, **usage.as_update()}

    def _execute(self, code: str, usage: Usage) -> str:
        result = get_tool("python_exec")(code=code)
        if not result.ok:
            usage.error("analyst.python_exec", result.error)
            return f"ERROR: {result.error}"
        data = result.data or {}
        return f"stdout:\n{data.get('stdout', '')}\nresult: {data.get('result')}"[:MAX_OUTPUT_CHARS]

    def _lessons(self, state: DeskState) -> tuple[str, list[str]]:
        hits = self._services.knowledge.lessons.relevant(
            f"{state['ticker']} {state.get('sector') or ''} прогноз {state['query']}",
            as_of=as_of_from_iso(state.get("as_of_date")),
            ticker=state["ticker"], sector=state.get("sector"),
        )
        if not hits:
            return "Нет релевантных уроков.", []
        return "\n".join(f"- {h.text}" for h in hits), [h.id for h in hits]

