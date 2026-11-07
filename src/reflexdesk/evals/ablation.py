"""Ablation-протокол: single -> +debate -> full на одном golden-set."""

from __future__ import annotations

from typing import Any

from reflexdesk.evals.golden import run_golden

ORDER = ("single", "debate", "full")
COLUMNS = ("hit_rate", "faithfulness", "grounded", "judge", "rejected_round1", "cost_usd", "latency_s")


def run_ablation(limit: int | None = None) -> dict[str, Any]:
    """Таблица сравнения конфигураций контура."""
    rows = {config: run_golden(config, limit=limit)["summary"] for config in ORDER}
    table = [{"config": c, **{k: rows[c].get(k) for k in COLUMNS}} for c in ORDER]
    return {"table": table}
