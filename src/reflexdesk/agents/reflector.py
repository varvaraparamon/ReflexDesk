"""Reflector (off-line): сверка созревших прогнозов с рынком и извлечение уроков через LangMem в L3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

import openai
from langmem import create_memory_manager

from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import REFLECTOR
from reflexdesk.evals.metrics import brier, hit_rate, is_hit
from reflexdesk.obs import metrics
from reflexdesk.obs.context import bind
from reflexdesk.obs.logging import get_logger
from reflexdesk.obs.tracing import observe
from reflexdesk.router.langchain import chat_model
from reflexdesk.router.results import TierUnavailable
from reflexdesk.schemas import Lesson, LessonContent, Prediction, utcnow
from reflexdesk.tools.base import ToolError
from reflexdesk.tools.market import total_return

log = get_logger("reflector")

EXPIRE_AFTER = timedelta(days=14)
REFLECTOR_RUN_ID = "reflector"


@dataclass
class ReflectionReport:
    """Итог прогона Reflector."""

    verified: dict[str, int] = field(default_factory=dict)
    lessons_added: int = 0
    lessons_superseded: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


class Reflector:
    """Контур рефлексии; не участвует в онлайн-прогоне."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._agent = services.agent(REFLECTOR)

    def run(self, now=None) -> ReflectionReport:
        """Сверяет созревшие прогнозы и пишет уроки по каждому тикеру."""
        now = now or utcnow()
        report = ReflectionReport()
        with bind(run_id=f"{REFLECTOR_RUN_ID}-{now:%Y%m%d%H%M}", node="reflector"), observe("reflector", "agent"):
            by_ticker: dict[str, list[Prediction]] = defaultdict(list)
            for prediction in self._services.episodic.due_predictions(now):
                checked = self._verify(prediction, now, report)
                if checked is not None:
                    by_ticker[checked.ticker].append(checked)
            for ticker, items in by_ticker.items():
                self._reflect(ticker, items, report)
        self.publish_metrics(now)
        return report

    def _verify(self, prediction: Prediction, now, report: ReflectionReport) -> Prediction | None:
        try:
            actual = total_return(prediction.ticker, prediction.created_at.date(), prediction.check_at.date())
        except (ToolError, ValueError, KeyError) as exc:
            if now - prediction.check_at > EXPIRE_AFTER:
                self._services.episodic.set_status(prediction.prediction_id, "expired", None)
                report.verified["expired"] = report.verified.get("expired", 0) + 1
            else:
                report.errors.append(f"{prediction.prediction_id}: {exc}")
            return None
        status = "hit" if is_hit(prediction.direction, actual) else "miss"
        self._services.episodic.set_status(prediction.prediction_id, status, actual)
        report.verified[status] = report.verified.get(status, 0) + 1
        return prediction.model_copy(update={"status": status, "actual_return": actual, "verified_at": now})

    def _trajectory(self, prediction: Prediction) -> str:
        theses = [
            d["artifact"].get("analysis_draft", {}).get("theses", [])
            for d in self._services.episodic.decisions(prediction.run_id, node="analyst")
        ]
        return (
            f"Прогноз {prediction.prediction_id}: {prediction.ticker} {prediction.direction}, горизонт "
            f"{prediction.horizon_days} дн, уверенность {prediction.confidence}, дата {prediction.created_at.date()}.\n"
            f"Тезис: {prediction.thesis}\nТезисы аналитика: {theses[-1] if theses else 'н/д'}\n"
            f"Факт: доходность {prediction.actual_return:+.2%} к {prediction.check_at.date()}, статус {prediction.status}."
        )

    def _reflect(self, ticker: str, predictions: list[Prediction], report: ReflectionReport) -> None:
        knowledge = self._services.knowledge
        existing = [(row["id"], LessonContent.model_validate(row["lesson"])) for row in knowledge.lessons.active(ticker)]
        messages = [{"role": "user", "content": "\n\n".join(self._trajectory(p) for p in predictions)}]
        extracted = self._extract(messages, existing, report)
        existing_ids = {lesson_id for lesson_id, _ in existing}
        sector = (knowledge.profiles.get(ticker) or {}).get("sector")
        source = predictions[0].prediction_id if len(predictions) == 1 else None
        new_lessons = []
        for item in extracted:
            content = item.content
            if not isinstance(content, LessonContent):
                continue
            lesson = Lesson(**{**content.model_dump(), "sector": content.sector or sector}, prediction_id=source)
            new_lessons.append(lesson)
            if item.id in existing_ids:
                knowledge.lessons.supersede(item.id, lesson.lesson_id)
                report.lessons_superseded += 1
        knowledge.lessons.add(new_lessons)
        report.lessons_added += len(new_lessons)

    def _extract(self, messages, existing, report: ReflectionReport):
        """LangMem memory manager по цепочке тиров роутера (T3 -> T2); недоступность всей цепочки - ошибка."""
        router = self._services.router
        last_error: Exception | None = None
        for tier in router.fallback_chain(REFLECTOR.tier):
            model, callback = chat_model(router, tier)
            manager = create_memory_manager(
                model, schemas=[LessonContent], instructions=self._agent.prompt.text,
                enable_inserts=True, enable_updates=True, enable_deletes=False,
            )
            try:
                result = manager.invoke({"messages": messages, "existing": existing})
                report.cost_usd += callback.cost_usd
                return result
            except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError) as exc:
                report.cost_usd += callback.cost_usd
                last_error = exc
                log.warning("lesson extraction tier unavailable", tier=tier, error=str(exc))
        raise TierUnavailable(f"reflector: all tiers failed: {last_error}")

    def publish_metrics(self, now=None) -> dict[str, float | None]:
        """Счётчики pending/stale и продуктовые метрики в Prometheus."""
        episodic = self._services.episodic
        counts = episodic.counts(now)
        metrics.predictions_pending.set(counts["pending"])
        metrics.predictions_stale.set(counts["stale"])
        everything = episodic.predictions(working_only=True)
        values = {"hit_rate": hit_rate(everything), "brier": brier(everything)}
        for name, value in values.items():
            if value is not None:
                metrics.eval_score.labels("self_eval", name).set(value)
        return values
