"""Еженедельный бэктест сверенных прогнозов против бенчмарка (стратегия деска vs buy-and-hold)."""

from __future__ import annotations

import csv
from typing import Any

from reflexdesk.config import get_settings
from reflexdesk.evals.metrics import verified
from reflexdesk.memory import get_episodic
from reflexdesk.obs import metrics
from reflexdesk.schemas import utcnow
from reflexdesk.tools import get_tool


def signals_from_predictions(predictions) -> list[dict[str, Any]]:
    """Прогнозы -> сигналы бэктеста на интервале [created_at, check_at]."""
    return [
        {"ticker": p.ticker, "direction": p.direction, "start": p.created_at.date().isoformat(), "end": p.check_at.date().isoformat()}
        for p in predictions
    ]


def run_backtest_report(benchmark: str = "SPY") -> dict[str, Any]:
    """Запускает run_backtest, пишет кривую капитала в CSV и метрики в Prometheus."""
    signals = signals_from_predictions(verified(get_episodic().predictions(working_only=True)))
    if not signals:
        return {"error": "нет сверенных прогнозов"}
    result = get_tool("run_backtest")(signals=signals, benchmark=benchmark)
    if not result.ok:
        return {"error": result.error}
    data = dict(result.data)
    curve = data.pop("equity_curve", {})
    path = get_settings().data_dir / "evals" / f"equity_{utcnow():%Y%m%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "desk", benchmark])
        writer.writerows([d, *values] for d, values in curve.items())
    for side in ("strategy", "benchmark"):
        for name, value in data[side].items():
            metrics.eval_score.labels(f"backtest_{side}", name).set(value)
    return {**data, "equity_curve_path": str(path)}
