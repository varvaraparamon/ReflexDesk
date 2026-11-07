"""Pydantic-артефакты агентов и записи памяти."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

Direction = Literal["up", "down", "flat"]
PredictionStatus = Literal["pending", "hit", "miss", "expired"]
Severity = Literal["low", "medium", "high"]
HORIZONS = (30, 90, 180)


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """Наивный datetime трактуется как UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


class SubTask(BaseModel):
    """Подзадача плана Orchestrator."""

    assignee: Literal["collector", "analyst"]
    goal: str
    hints: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """План прогона: тикер и подзадачи."""

    ticker: str | None = None
    tickers: list[str] = Field(default_factory=list)
    subtasks: list[SubTask]
    rationale: str = ""


class Evidence(BaseModel):
    """Факт с обязательным источником."""

    fact: str
    value: str | float | None = None
    source_url: str
    retrieved_at: datetime = Field(default_factory=utcnow)
    published_at: datetime | None = None
    tool: str = ""

    @field_validator("source_url")
    @classmethod
    def _source_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_url is required: факт без источника не существует")
        return v.strip()


class EvidenceBatch(BaseModel):
    """Выход шага Data Collector."""

    evidence: list[Evidence]


class DraftPrediction(BaseModel):
    """Черновой прогноз аналитика."""

    ticker: str
    direction: Direction
    horizon_days: int
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str

    @field_validator("horizon_days")
    @classmethod
    def _allowed_horizon(cls, v: int) -> int:
        if v not in HORIZONS:
            raise ValueError(f"horizon_days must be one of {HORIZONS}")
        return v


class Valuation(BaseModel):
    """Оценка: метод, диапазон, допущения."""

    method: str
    low: float | None = None
    base: float | None = None
    high: float | None = None
    assumptions: list[str] = Field(default_factory=list)


class AnalysisDraft(BaseModel):
    """Артефакт Analyst: тезисы, оценки, черновые прогнозы."""

    ticker: str
    theses: list[str]
    key_metrics: dict[str, str] = Field(default_factory=dict)
    valuations: list[Valuation] = Field(default_factory=list)
    predictions: list[DraftPrediction] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    computations: list[str] = Field(default_factory=list)


class AnalystStep(BaseModel):
    """Шаг Analyst: либо код для python_exec, либо финальный драфт."""

    code: str | None = None
    draft: AnalysisDraft | None = None

    @model_validator(mode="after")
    def _one_of(self) -> AnalystStep:
        if (self.code is None) == (self.draft is None):
            raise ValueError("exactly one of code/draft must be set")
        return self


class DebateCase(BaseModel):
    """Позиция Bull или Bear."""

    side: Literal["bull", "bear"]
    arguments: list[str] = Field(min_length=3, max_length=3)
    weakest_point: str
    summary: str


class RiskFlag(BaseModel):
    type: str
    severity: Severity
    note: str


class RiskReport(BaseModel):
    """Артефакт Risk Officer."""

    flags: list[RiskFlag] = Field(default_factory=list)
    summary: str = ""


class FailedClaim(BaseModel):
    claim: str
    reason: str


class ValidationResult(BaseModel):
    """Артефакт Validator."""

    verdict: Literal["ok", "fail"]
    failed_claims: list[FailedClaim] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    notes: str = ""
    skipped: bool = False
    degraded: bool = False


class Prediction(BaseModel):
    """Прогноз в журнале L2."""

    prediction_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    ticker: str
    direction: Direction
    horizon_days: int
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    created_at: datetime = Field(default_factory=utcnow)
    check_at: datetime | None = None
    status: PredictionStatus = "pending"
    actual_return: float | None = None
    verified_at: datetime | None = None

    @model_validator(mode="after")
    def _derive_check_at(self) -> Prediction:
        if self.check_at is None:
            self.check_at = self.created_at + timedelta(days=self.horizon_days)
        return self

    @classmethod
    def from_draft(cls, draft: DraftPrediction, run_id: str, created_at: datetime) -> Prediction:
        return cls(run_id=run_id, created_at=created_at, **draft.model_dump())


class LessonContent(BaseModel):
    """Урок Reflector: контекст → ожидание → факт → причина → правило."""

    ticker: str
    sector: str | None = None
    error_type: Literal["overconfidence", "wrong_direction", "ignored_risk", "data_gap", "timing", "correct_call"]
    context: str
    expectation: str
    outcome: str
    cause: str
    rule: str


class Lesson(LessonContent):
    """Урок в L3 со служебными полями."""

    lesson_id: str = Field(default_factory=lambda: str(uuid4()))
    prediction_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    superseded_by: str | None = None

    @property
    def text(self) -> str:
        return (
            f"[{self.ticker}/{self.error_type}] Контекст: {self.context}. Ожидание: {self.expectation}. "
            f"Факт: {self.outcome}. Причина: {self.cause}. Правило: {self.rule}"
        )


def as_of_datetime(as_of: date | None) -> datetime:
    """Момент среза данных: конец дня as_of или текущее время."""
    if as_of is None:
        return utcnow()
    return datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=UTC)


def as_of_from_iso(value: str | None) -> datetime:
    """Срез данных из ISO-даты прогона (None — сейчас)."""
    return as_of_datetime(date.fromisoformat(value) if value else None)
