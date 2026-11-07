"""Ночной golden-set с регрессионным гейтом judge-score (> 0.5 балла) и еженедельный бэктест."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scheduling import run_daily

from reflexdesk.config import get_settings
from reflexdesk.evals.backtest import run_backtest_report
from reflexdesk.evals.golden import run_golden
from reflexdesk.obs import metrics
from reflexdesk.obs.logging import get_logger
from reflexdesk.schemas import utcnow

log = get_logger("eval_runner")

REGRESSION_DROP = 0.5
NIGHTLY_HOUR_UTC = 2


def previous_judge(config: str, exclude: str) -> float | None:
    """Judge-score предыдущего прогона golden той же конфигурации."""
    files = sorted((get_settings().data_dir / "evals").glob(f"golden_{config}-*.json"))
    for path in reversed(files):
        if path.name == Path(exclude).name:
            continue
        return json.loads(path.read_text(encoding="utf-8"))["summary"].get("judge")
    return None


def nightly(config: str = "full", limit: int | None = None) -> int:
    """Возвращает код выхода: 1 при регрессии judge-score."""
    result = run_golden(config, limit=limit)
    current, previous = result["summary"].get("judge"), previous_judge(config, result["path"])
    regression = current is not None and previous is not None and previous - current > REGRESSION_DROP
    metrics.eval_score.labels(f"golden_{config}", "regression").set(float(regression))
    log.info("golden done", judge=current, previous=previous, regression=regression)
    if utcnow().weekday() == 0:
        log.info("weekly backtest", result=json.dumps(run_backtest_report(), default=str))
    return 1 if regression else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--config", default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metrics-port", type=int, default=9102)
    args = parser.parse_args()
    if args.once:
        sys.exit(nightly(args.config, args.limit))
    run_daily(lambda: nightly(args.config, args.limit), NIGHTLY_HOUR_UTC, args.metrics_port)
