"""Golden-set: 30 исторических кейсов (без look-ahead); faithfulness, judge, hit rate, cost — в Langfuse dataset run."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from reflexdesk.agents.base import Usage
from reflexdesk.config import get_settings
from reflexdesk.evals.judge import judge_report
from reflexdesk.evals.metrics import is_hit
from reflexdesk.graph import CONFIGS
from reflexdesk.memory.episodic import EVAL_TAG_PREFIX
from reflexdesk.obs import metrics
from reflexdesk.obs.context import bind
from reflexdesk.obs.tracing import run_experiment
from reflexdesk.reporting.claims import computed_numbers, faithfulness
from reflexdesk.schemas import utcnow
from reflexdesk.service import DeskService
from reflexdesk.tools.base import ToolError
from reflexdesk.tools.market import total_return

CASES_PATH = Path(__file__).with_name("cases.yaml")
DATASET = "golden"


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    return cases[:limit] if limit else cases


def historical_hits(predictions) -> list[bool]:
    """Исход прогнозов на исторических датах без мутации L2."""
    hits = []
    for p in predictions:
        try:
            hits.append(is_hit(p.direction, total_return(p.ticker, p.created_at.date(), p.check_at.date())))
        except (ToolError, ValueError, KeyError):
            continue
    return hits


class GoldenSuite:
    """Task и evaluators golden-set для одной конфигурации контура."""

    def __init__(self, config: str) -> None:
        self.config = config
        self.service = DeskService(config=CONFIGS[config])

    def task(self, case: dict[str, Any]) -> dict[str, Any]:
        final = self.service.analyze(case["query"], date.fromisoformat(case["as_of"]), tags=f"{EVAL_TAG_PREFIX}golden:{self.config}")
        return {"run_id": final["run_id"], "status": final.get("status"), "cost_usd": final.get("cost_usd", 0.0)}

    def evaluate(self, case: dict[str, Any], output: dict[str, Any]) -> dict[str, float]:
        run_id = output["run_id"]
        state = self.service.state(run_id)
        episodic = self.service.services.episodic
        completed = output.get("status") == "done"
        scores: dict[str, float] = {"cost_usd": float(output["cost_usd"] or 0.0), "completed": float(completed)}
        report = self.service.report(run_id) if completed else None
        if report is None:
            return scores
        markdown, _ = report
        evidence = state.get("evidence", [])
        draft = state.get("analysis_draft")
        scores["faithfulness"] = faithfulness(markdown, evidence).ratio
        scores["grounded"] = faithfulness(markdown, evidence, computed_numbers(draft.computations if draft else [])).grounded
        first_try = state.get("json_first_try", [])
        if first_try:
            scores["json_first_try"] = sum(first_try) / len(first_try)
        validator = episodic.decisions(run_id, node="validator")
        if validator:
            scores["rejected_round1"] = float(validator[0]["artifact"]["validation"]["verdict"] == "fail")
        hits = historical_hits(episodic.predictions(run_id=run_id))
        if hits:
            scores["hit_rate"] = sum(hits) / len(hits)
        with bind(run_id=run_id):
            judged = judge_report(self.service.services, markdown, Usage())
        if judged is not None:
            scores["judge"] = judged.mean
        outcome = episodic.outcome(run_id)
        if outcome:
            scores["latency_s"] = float(outcome["latency_s"] or 0.0)
        return scores


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    """Средние по всем метрикам кейсов."""
    keys = {k for r in results for k in r["scores"]}
    return {k: round(mean(r["scores"][k] for r in results if k in r["scores"]), 4) for k in sorted(keys)}


def run_golden(config: str = "full", limit: int | None = None) -> dict[str, Any]:
    """Прогон golden-set; результаты в Langfuse, Prometheus и data/evals/*.json."""
    suite = GoldenSuite(config)
    run_name = f"{config}-{utcnow():%Y%m%dT%H%M}"
    results = run_experiment(DATASET, run_name, load_cases(limit), suite.task, suite.evaluate)
    summary = aggregate(results) if results else {}
    for name, value in summary.items():
        metrics.eval_score.labels(f"golden_{config}", name).set(value)
    out = get_settings().data_dir / "evals" / f"golden_{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_name": run_name, "config": config, "summary": summary, "cases": results}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"run_name": run_name, "config": config, "cases": len(results), "summary": summary, "path": str(out)}
