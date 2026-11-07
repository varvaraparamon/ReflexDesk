"""Схема состояния графа DeskState."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from reflexdesk.schemas import (
    AnalysisDraft,
    DebateCase,
    Evidence,
    Plan,
    RiskReport,
    ValidationResult,
)


def merge_unique(left: list[str], right: list[str]) -> list[str]:
    """Объединение списков без дублей с сохранением порядка."""
    return list(dict.fromkeys([*left, *right]))


class DeskState(TypedDict, total=False):
    """Состояние прогона; поля с редьюсерами безопасны для параллельных веток."""

    run_id: str
    query: str
    as_of_date: str | None
    scenario: str
    ticker: str | None
    plan: Plan | None
    evidence: Annotated[list[Evidence], operator.add]
    analysis_draft: AnalysisDraft | None
    risk_score: int | None
    bull_case: DebateCase | None
    bear_case: DebateCase | None
    risk_report: RiskReport | None
    report_draft: str | None
    validation: ValidationResult | None
    validation_round: int
    lessons_used: Annotated[list[str], merge_unique]
    cost_usd: Annotated[float, operator.add]
    errors: Annotated[list[str], operator.add]
    flags: Annotated[list[str], merge_unique]
    gate_decisions: Annotated[list[dict], operator.add]
    json_first_try: Annotated[list[bool], operator.add]
    sector: str | None
    report_path: str | None
    pdf_path: str | None
    prediction_ids: list[str]
    status: str


def initial_state(run_id: str, query: str, as_of_date: str | None) -> DeskState:
    """Начальное состояние прогона."""
    return DeskState(
        run_id=run_id,
        query=query,
        as_of_date=as_of_date,
        scenario="",
        ticker=None,
        plan=None,
        evidence=[],
        analysis_draft=None,
        risk_score=None,
        bull_case=None,
        bear_case=None,
        risk_report=None,
        report_draft=None,
        validation=None,
        validation_round=0,
        lessons_used=[],
        cost_usd=0.0,
        errors=[],
        flags=[],
        gate_decisions=[],
        json_first_try=[],
        sector=None,
        report_path=None,
        pdf_path=None,
        prediction_ids=[],
        status="running",
    )
