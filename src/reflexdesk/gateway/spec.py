"""Контракты Jev-like слоя решений."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DecisionKind = Literal["choice", "noul", "score"]

NOUL_OPTIONS = ["yes", "no"]
SCORE_OPTIONS = [str(i) for i in range(11)]
MAX_CONTEXT_CHARS = 6000
MAX_OPTIONS = 26


class DecisionSpec(BaseModel):
    """Запрос на решение: вид, состояние (контекст), вопрос, варианты и их описания."""

    kind: DecisionKind
    context: str
    question: str
    options: list[str] = Field(default_factory=list)
    criteria: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> DecisionSpec:
        if self.kind == "noul":
            self.options = list(NOUL_OPTIONS)
        elif self.kind == "score" and not self.options:
            self.options = list(SCORE_OPTIONS)
        if not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"decision needs 2..{MAX_OPTIONS} options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        self.context = self.context[-MAX_CONTEXT_CHARS:]
        return self


class Decision(BaseModel):
    """Результат решения: распределение по вариантам и выбор."""

    kind: str
    distribution: dict[str, float]
    pick: str
    confidence: float
    latency_ms: int
    gate: str = ""
    jev_fallback: bool = False
    route: Literal["one_forward", "cot"] = "one_forward"
    spec: DecisionSpec | None = None

    @property
    def expected_score(self) -> float:
        """Матожидание для score-решений."""
        return sum(float(k) * p for k, p in self.distribution.items())
