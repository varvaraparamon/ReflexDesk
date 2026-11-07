"""Выпуск результата прогона: отчёт md/pdf, прогнозы из таблицы отчёта в L2, исход, профиль тикера в L3."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from reflexdesk.agents.services import Services
from reflexdesk.obs import metrics
from reflexdesk.reporting.parse import parse_predictions
from reflexdesk.schemas import (
    DraftPrediction,
    Prediction,
    ValidationResult,
    as_of_from_iso,
    ensure_utc,
    utcnow,
)
from reflexdesk.state import DeskState
from reflexdesk.tools import get_tool

UNVALIDATED_BANNER = "> **UNVALIDATED**: отчёт не прошёл валидацию за допустимое число раундов; числа требуют ручной проверки.\n\n"
REJECT_TEXT = (
    "# Запрос отклонён\n\nЗапрос вне компетенции research-деска (не про инвестиционный анализ "
    "или требует исполнения сделок). Деск анализирует тикеры и портфели и не даёт персональных советов.\n"
)
NO_STANCE = "нет позиции"


class ReportGenerationError(RuntimeError):
    """Отчёт не удалось сохранить (md/pdf)."""


def prediction_id(run_id: str, index: int, draft: DraftPrediction) -> str:
    """Детерминированный id прогноза — повторный finalize после сбоя не дублирует записи."""
    return hashlib.sha256(f"{run_id}:{index}:{draft.ticker}:{draft.horizon_days}".encode()).hexdigest()[:32]


def is_unvalidated(state: DeskState) -> bool:
    """Валидация запускалась, но не дала verdict=ok моделью другого семейства (деградация T3→T2 не засчитывается)."""
    validation: ValidationResult | None = state.get("validation")
    passed = validation is not None and validation.verdict == "ok" and not validation.degraded
    return state.get("validation_round", 0) > 0 and not passed


class FinalizeNode:
    """Финальный узел: идемпотентен относительно L2."""

    def __init__(self, services: Services) -> None:
        self._services = services

    def __call__(self, state: DeskState) -> dict[str, Any]:
        status = state.get("status", "running")
        if status == "rejected":
            return self._emit(state, REJECT_TEXT, status="rejected", predictions=[])
        if state.get("analysis_draft") is None or not state.get("report_draft"):
            text = "# Прогон не завершён\n\nОшибки:\n" + "\n".join(f"- {e}" for e in state.get("errors", []))
            return self._emit(state, text, status="failed", predictions=[])

        markdown = state["report_draft"]
        flags: list[str] = []
        errors: list[str] = []
        if is_unvalidated(state):
            markdown = UNVALIDATED_BANNER + markdown
            flags.append("UNVALIDATED")
            metrics.validation_exhausted_total.inc()
        predictions, table_errors = parse_predictions(markdown)
        errors += [f"finalize.predictions: {e}" for e in table_errors]
        return self._emit(state, markdown, status="done", predictions=predictions, flags=flags, errors=errors)

    def _emit(self, state: DeskState, markdown: str, *, status: str, predictions: list[DraftPrediction],
              flags: list[str] | None = None, errors: list[str] | None = None) -> dict[str, Any]:
        run_id = state["run_id"]
        episodic = self._services.episodic
        rendered = get_tool("generate_report")(markdown=markdown)
        if not rendered.ok:
            raise ReportGenerationError(rendered.error)
        created_at = as_of_from_iso(state.get("as_of_date")) if state.get("as_of_date") else utcnow()
        records = [
            Prediction(prediction_id=prediction_id(run_id, i, d), run_id=run_id, created_at=created_at, **d.model_dump())
            for i, d in enumerate(predictions)
        ]
        known = {p.prediction_id for p in episodic.predictions(run_id=run_id)}
        episodic.add_predictions([p for p in records if p.prediction_id not in known])
        if episodic.outcome(run_id) is None:
            started = ensure_utc(episodic.run(run_id)["created_at"])
            episodic.record_outcome(
                run_id, rendered.data["md_path"], state.get("validation"), state.get("cost_usd", 0.0),
                (utcnow() - started).total_seconds(),
            )
        if status == "done":
            self._update_profile(state, records)
        return {
            "status": status,
            "report_path": rendered.data["md_path"],
            "pdf_path": rendered.data["pdf_path"],
            "prediction_ids": [p.prediction_id for p in records],
            "flags": flags or [],
            "errors": errors or [],
        }

    def _update_profile(self, state: DeskState, predictions: list[Prediction]) -> None:
        stance = Counter(p.direction for p in predictions).most_common(1)[0][0] if predictions else NO_STANCE
        self._services.knowledge.profiles.update(
            state["ticker"], sector=state.get("sector"), metrics=state["analysis_draft"].key_metrics,
            stance=stance, as_of=as_of_from_iso(state.get("as_of_date")),
        )
