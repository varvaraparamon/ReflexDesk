"""Prometheus-метрики деска."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
RUN_BUCKETS = (30, 60, 120, 180, 300, 450, 600, 720, 900, 1200)

llm_latency_seconds = Histogram("llm_latency_seconds", "Латентность LLM", ["tier"], buckets=LATENCY_BUCKETS)
llm_cost_usd_total = Counter("llm_cost_usd_total", "Расход на LLM, USD", ["tier"])
llm_tokens_total = Counter("llm_tokens_total", "Токены LLM", ["tier", "kind"])

gateway_decisions_total = Counter("gateway_decisions_total", "Решения gateway", ["gate", "pick"])
gateway_latency_seconds = Histogram(
    "gateway_latency_seconds", "Латентность решения gateway", ["gate"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2),
)
jev_fallback_total = Counter("jev_fallback_total", "Решения gateway, переданные T0", ["gate"])

validation_rejected_total = Counter("validation_rejected_total", "Черновики, отклонённые Validator")
validation_exhausted_total = Counter("validation_exhausted_total", "Прогоны, выпущенные UNVALIDATED")

tool_calls_total = Counter("tool_calls_total", "Вызовы тулов", ["tool", "status"])

graph_run_duration_seconds = Histogram("graph_run_duration_seconds", "Длительность прогона", buckets=RUN_BUCKETS)
graph_runs_total = Counter("graph_runs_total", "Прогоны графа", ["scenario", "status"])

predictions_pending = Gauge("predictions_pending", "Прогнозы в ожидании сверки")
predictions_stale = Gauge("predictions_stale", "Просроченные >2 дней pending-прогнозы")

eval_score = Gauge("eval_score", "Последнее значение eval-метрики", ["suite", "metric"])
