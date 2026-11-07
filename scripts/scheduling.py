"""Планировщик ежедневных задач с экспортом метрик Prometheus"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from prometheus_client import start_http_server

from reflexdesk.obs.logging import get_logger

log = get_logger("scheduler")


def seconds_until(hour_utc: int, now: datetime | None = None) -> float:
    """Секунды до ближайшего наступления hour_utc:00 UTC."""
    now = now or datetime.now(UTC)
    target = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_daily(job: Callable[[], None], hour_utc: int, metrics_port: int | None = None) -> None:
    """Бесконечный цикл: job раз в сутки в hour_utc."""
    if metrics_port:
        start_http_server(metrics_port)
    while True:
        delay = seconds_until(hour_utc)
        log.info("next run scheduled", in_s=int(delay))
        time.sleep(delay)
        try:
            job()
        except Exception:
            log.exception("scheduled job failed")
