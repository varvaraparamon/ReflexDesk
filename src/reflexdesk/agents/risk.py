"""Риск-гейт G5 и Risk Officer."""

from __future__ import annotations

from typing import Any

from reflexdesk.agents.base import EvidenceView, Usage
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import RISK_OFFICER
from reflexdesk.schemas import RiskReport
from reflexdesk.state import DeskState


class RiskGateNode:
    """G5: score 0–10 рискованности драфта."""

    def __init__(self, services: Services) -> None:
        self._gates = services.gates

    def __call__(self, state: DeskState) -> dict[str, Any]:
        draft = state["analysis_draft"]
        decision = self._gates.risk_score(draft.model_dump_json(exclude={"computations"}))
        return {"risk_score": int(decision.pick), "gate_decisions": [decision.model_dump()]}


class RiskOfficerNode:
    """Чек-лист рисков по драфту и Evidence."""

    def __init__(self, services: Services) -> None:
        self._agent = services.agent(RISK_OFFICER)

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        draft = state["analysis_draft"]
        view = EvidenceView(state.get("evidence", []))
        prompt = (
            f"Дата среза: {state.get('as_of_date') or 'сегодня'}. Риск-скор G5: {state.get('risk_score')}.\n"
            f"Аналитический драфт:\n{draft.model_dump_json(exclude={'computations'})}\n\n{view.render()}"
        )
        report = self._agent.ask(prompt, RiskReport, usage)
        return {"risk_report": report, **usage.as_update()}
