"""Агрегаты для /v1/metrics/summary: hit rate, Brier, калибровка, стоимость и латентность отчётов."""

from __future__ import annotations

import json
from dataclasses import asdict
from statistics import median
from typing import Any

from reflexdesk.evals.metrics import brier, calibration, hit_rate
from reflexdesk.memory.episodic import EpisodicStore


def percentile(values: list[float], q: float) -> float | None:
    """Перцентиль по ближайшему рангу."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)]


def metrics_summary(store: EpisodicStore) -> dict[str, Any]:
    """Сводка продуктовых и инфраструктурных метрик из L2."""
    predictions = store.predictions(working_only=True)
    outcomes = store.outcomes(working_only=True)
    costs = [o["cost_usd"] or 0.0 for o in outcomes]
    latencies = [o["latency_s"] or 0.0 for o in outcomes]
    verdicts = [json.loads(o["validation_json"] or "null") for o in outcomes]
    validated = [v for v in verdicts if v]
    return {
        "predictions": {
            "total": len(predictions),
            "by_status": {s: sum(p.status == s for p in predictions) for s in ("pending", "hit", "miss", "expired")},
            "hit_rate": hit_rate(predictions),
            "brier": brier(predictions),
            "calibration": [asdict(b) for b in calibration(predictions)],
        },
        "reports": {
            "count": len(outcomes),
            "cost_usd_mean": sum(costs) / len(costs) if costs else None,
            "cost_usd_median": median(costs) if costs else None,
            "latency_s_p50": percentile(latencies, 0.5),
            "latency_s_p95": percentile(latencies, 0.95),
            "validation_ok_share": sum(v.get("verdict") == "ok" for v in validated) / len(validated) if validated else None,
        },
    }
