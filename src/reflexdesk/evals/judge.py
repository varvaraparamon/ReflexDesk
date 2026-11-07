"""T3-judge: оценка отчёта по рубрике 0–10 × 4 оси."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reflexdesk.agents.base import AgentSpec, Usage
from reflexdesk.agents.services import Services

JUDGE = AgentSpec("judge", "tier3", documents=("HOUSE_STYLE.md",))


class JudgeScore(BaseModel):
    """Оценки по осям рубрики."""

    completeness: int = Field(ge=0, le=10)
    evidence: int = Field(ge=0, le=10)
    reasoning: int = Field(ge=0, le=10)
    style: int = Field(ge=0, le=10)
    comment: str = ""

    @property
    def mean(self) -> float:
        return (self.completeness + self.evidence + self.reasoning + self.style) / 4


def judge_report(services: Services, markdown: str, usage: Usage) -> JudgeScore | None:
    """Оценка отчёта кросс-модельным судьёй."""
    return services.agent(JUDGE).ask(f"Отчёт для оценки:\n\n{markdown}", JudgeScore, usage)
