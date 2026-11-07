"""Report Composer: Markdown-отчёт по house style с учётом фидбэка Validator."""

from __future__ import annotations

import re
from typing import Any

from reflexdesk.agents.base import EvidenceView, Usage, to_json
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import COMPOSER
from reflexdesk.state import DeskState

_FENCE = re.compile(r"^```(?:markdown|md)?\s*\n|\n```\s*$")


class ComposerNode:
    """Сборка отчёта; на повторных раундах исправляет замечания валидатора."""

    def __init__(self, services: Services) -> None:
        self._agent = services.agent(COMPOSER)

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        view = EvidenceView(state.get("evidence", []))
        parts = [
            f"Тикер: {state['ticker']}. Дата среза: {state.get('as_of_date') or 'сегодня'}.",
            f"Аналитический драфт:\n{state['analysis_draft'].model_dump_json(exclude={'computations'})}",
            f"Позиция Bull:\n{to_json(state.get('bull_case')) if state.get('bull_case') else 'нет'}",
            f"Позиция Bear:\n{to_json(state.get('bear_case')) if state.get('bear_case') else 'нет'}",
            f"Отчёт о рисках:\n{to_json(state.get('risk_report')) if state.get('risk_report') else 'Risk Officer не запускался'}",
            view.render(),
        ]
        validation = state.get("validation")
        if validation is not None and validation.verdict == "fail" and state.get("report_draft"):
            parts.append(f"Предыдущий черновик:\n{state['report_draft']}")
            parts.append(f"Замечания валидатора (исправь каждое):\n{validation.model_dump_json()}")
        markdown = self._agent.write("\n\n".join(parts), usage)
        return {"report_draft": _FENCE.sub("", markdown.strip()), **usage.as_update()}
