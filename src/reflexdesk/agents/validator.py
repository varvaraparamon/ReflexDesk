"""Пре-фильтр G4 и Validator (T3): сверка черновика с Evidence + детерминированные проверки."""

from __future__ import annotations

from typing import Any

from reflexdesk.agents.base import EvidenceView, Usage
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import VALIDATOR
from reflexdesk.obs import metrics
from reflexdesk.reporting.claims import faithfulness
from reflexdesk.reporting.parse import parse_predictions
from reflexdesk.schemas import FailedClaim, ValidationResult
from reflexdesk.state import DeskState

DEGRADED_FLAG = "degraded_validation"
MAX_HINTS = 25


class ValidationGateNode:
    """G4: если проверяемых утверждений нет - Validator пропускается."""

    def __init__(self, services: Services) -> None:
        self._gates = services.gates

    def __call__(self, state: DeskState) -> dict[str, Any]:
        decision = self._gates.needs_validation(state["report_draft"] or "")
        update: dict[str, Any] = {"gate_decisions": [decision.model_dump()]}
        if decision.pick == "no":
            update["validation"] = ValidationResult(verdict="ok", skipped=True, notes="G4: нет проверяемых утверждений")
        return update


class ValidatorNode:
    """Кросс-модельная верификация: другая модель, reasoning-режим."""

    def __init__(self, services: Services) -> None:
        self._agent = services.agent(VALIDATOR)

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        draft = state["report_draft"] or ""
        evidence = state.get("evidence", [])
        computed = state["analysis_draft"].computations if state.get("analysis_draft") else []
        script = faithfulness(draft, evidence)
        _, table_errors = parse_predictions(draft)
        hints = "\n".join(f"- «{c.context}» (число {c.raw})" for c in script.unsupported[:MAX_HINTS]) or "нет"
        prompt = (
            f"Черновик отчёта:\n{draft}\n\n{EvidenceView(evidence).render()}\n\n"
            f"Результаты расчётов аналитика (python_exec):\n{chr(10).join(computed) or 'нет'}\n\n"
            f"Скрипт сверки не нашёл в Evidence следующие числа (проверь, не выводятся ли они из Evidence или расчётов):\n{hints}\n"
            f"Ошибки таблицы прогнозов: {'; '.join(table_errors) or 'нет'}"
        )
        result = self._agent.ask(prompt, ValidationResult, usage)
        if result is None:
            metrics.validation_rejected_total.inc()
            return {"validation": None, "validation_round": state.get("validation_round", 0) + 1, **usage.as_update()}
        if table_errors:
            result.verdict = "fail"
            result.failed_claims += [FailedClaim(claim="## Прогнозы", reason=e) for e in table_errors]
        result.degraded = DEGRADED_FLAG in usage.flags
        if result.verdict == "fail":
            metrics.validation_rejected_total.inc()
        return {"validation": result, "validation_round": state.get("validation_round", 0) + 1, **usage.as_update()}
