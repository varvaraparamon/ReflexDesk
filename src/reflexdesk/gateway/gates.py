"""Фиксированный набор гейтов G1–G5: спецификации вопросов к Jev-модели и решения через gateway.decide()."""

from __future__ import annotations

from reflexdesk.gateway.gateway import Gateway
from reflexdesk.gateway.spec import Decision, DecisionSpec

SCENARIOS = {
    "analyze_ticker": "analysis of one specific stock, ticker or company",
    "portfolio_qa": "question about a portfolio of several securities, allocation or diversification",
    "reject": "not about investing or markets, or a request to execute a trade (buy, sell, place an order)",
}

RETRY_CRITERIA = {
    "yes": "the error is caused by the arguments: misspelled ticker, wrong filing form, invalid or swapped dates",
    "no": "network or service failure (connection, timeout, 5xx), exhausted limit, disabled feature",
}

VALIDATION_CRITERIA = {
    "yes": "the draft states numbers, facts or forecasts that can be checked against sources",
    "no": "the draft has no checkable claims (refusal, error notice, general text without numbers)",
}

RISK_LEVELS = {
    "0": "no material risk: conservative forecast, complete and consistent data",
    "2": "low risk: moderate confidence, minor data gaps",
    "4": "moderate risk: some gaps or elevated volatility",
    "6": "elevated risk: notable data gaps, volatility or valuation stretch",
    "8": "high risk: aggressive forecast (confidence above 0.75 or large short-horizon move) or important data gaps",
    "10": "extreme risk: aggressive forecast combined with major data gaps or contradictory sources",
}

RISK_THRESHOLD = 7


class GateSpecs:
    """Спецификации гейтов — единый источник для онлайна и для оценки."""

    @staticmethod
    def route_query(query: str) -> DecisionSpec:
        return DecisionSpec(
            kind="choice", context=f"Request: {query}",
            question="Which scenario does this request belong to?",
            options=list(SCENARIOS), criteria=SCENARIOS,
        )

    @staticmethod
    def choose_tool(recent_steps: str, tools: dict[str, str]) -> DecisionSpec:
        return DecisionSpec(
            kind="choice", context=recent_steps,
            question=(
                "Which data collection tool should be called next? Choose finish only if every collection goal "
                "is already covered by a successfully called tool."
            ),
            options=list(tools), criteria=tools,
        )

    @staticmethod
    def should_retry(error: str, context: str) -> DecisionSpec:
        return DecisionSpec(
            kind="noul", context=f"Tool call: {context}\nError: {error}",
            question="Can this error be fixed by calling the tool again with different arguments?",
            criteria=RETRY_CRITERIA,
        )

    @staticmethod
    def needs_validation(draft: str) -> DecisionSpec:
        return DecisionSpec(
            kind="noul", context=draft,
            question="Does this report draft contain checkable claims: numbers, facts or forecasts?",
            criteria=VALIDATION_CRITERIA,
        )

    @staticmethod
    def risk_score(analysis: str) -> DecisionSpec:
        return DecisionSpec(
            kind="score", context=analysis,
            question="How risky is this analytical conclusion (0-10 scale)?",
            options=list(RISK_LEVELS), criteria=RISK_LEVELS,
        )


class Gates:
    """Типизированные обёртки над гейтами деска."""

    def __init__(self, gateway: Gateway) -> None:
        self._gateway = gateway

    def route_query(self, query: str) -> Decision:
        """G1: сценарий графа по тексту запроса."""
        return self._gateway.decide(GateSpecs.route_query(query), gate="G1")

    def choose_tool(self, recent_steps: str, tools: dict[str, str]) -> Decision:
        """G2: следующий инструмент Data Collector по последним шагам."""
        return self._gateway.decide(GateSpecs.choose_tool(recent_steps, tools), gate="G2")

    def should_retry(self, error: str, context: str) -> Decision:
        """G3: исправима ли ошибка тула повтором (yes) или нужна эскалация (no)."""
        return self._gateway.decide(GateSpecs.should_retry(error, context), gate="G3")

    def needs_validation(self, draft: str) -> Decision:
        """G4: содержит ли черновик проверяемые утверждения."""
        return self._gateway.decide(GateSpecs.needs_validation(draft), gate="G4")

    def risk_score(self, analysis: str) -> Decision:
        """G5: рискованность аналитического артефакта по шкале 0–10."""
        return self._gateway.decide(GateSpecs.risk_score(analysis), gate="G5")

    @staticmethod
    def is_risky(decision: Decision) -> bool:
        """Порог G5: score ≥ 7 требует Risk Officer."""
        return int(decision.pick) >= RISK_THRESHOLD
