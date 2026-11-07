"""Intake (гейт G1) и Orchestrator: сценарий прогона и план подзадач."""

from __future__ import annotations

from typing import Any

from reflexdesk.agents.base import Usage
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import ORCHESTRATOR
from reflexdesk.schemas import Plan
from reflexdesk.state import DeskState

REJECT = "reject"


class IntakeNode:
    """G1: маршрутизация запроса без генерации текста."""

    def __init__(self, services: Services) -> None:
        self._gates = services.gates

    def __call__(self, state: DeskState) -> dict[str, Any]:
        decision = self._gates.route_query(state["query"])
        update: dict[str, Any] = {"scenario": decision.pick, "gate_decisions": [decision.model_dump()]}
        if decision.pick == REJECT:
            update["status"] = "rejected"
        return update


class OrchestratorNode:
    """T2-планировщик: тикер(ы) и подзадачи для Collector/Analyst; без валидного плана прогон завершается."""

    def __init__(self, services: Services) -> None:
        self._agent = services.agent(ORCHESTRATOR)

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        prompt = (
            f"Запрос: {state['query']}\nСценарий: {state['scenario']}\n"
            f"Дата среза данных (as_of): {state.get('as_of_date') or 'сегодня'}"
        )
        plan = self._agent.ask(prompt, Plan, usage)
        if plan is None or not (plan.ticker or plan.tickers):
            usage.error("orchestrator", "план без тикера")
            return {"plan": None, "ticker": None, "status": "failed", **usage.as_update()}
        ticker = (plan.ticker or plan.tickers[0]).upper()
        plan.ticker = ticker
        plan.tickers = list(dict.fromkeys([ticker, *(t.upper() for t in plan.tickers)]))
        return {"plan": plan, "ticker": ticker, **usage.as_update()}
