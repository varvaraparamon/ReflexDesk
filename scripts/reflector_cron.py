"""Cron Reflector: 1 раз в день, раз в месяц - retention L1."""

from __future__ import annotations

import argparse
import json

from scheduling import run_daily

from reflexdesk.agents.reflector import Reflector
from reflexdesk.agents.services import default_services
from reflexdesk.memory.working import open_checkpointer, purge_expired
from reflexdesk.obs.logging import get_logger
from reflexdesk.schemas import utcnow

log = get_logger("reflector_cron")

OFF_PEAK_HOUR_UTC = 14


def job() -> None:
    services = default_services()
    report = Reflector(services).run()
    log.info("reflection done", report=json.dumps(report.__dict__, ensure_ascii=False, default=str))
    if utcnow().day == 1:
        purged = purge_expired(open_checkpointer(), services.episodic)
        log.info("checkpoint retention", purged=purged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--hour", type=int, default=OFF_PEAK_HOUR_UTC)
    parser.add_argument("--metrics-port", type=int, default=9101)
    args = parser.parse_args()
    if args.once:
        job()
    else:
        run_daily(job, args.hour, args.metrics_port)
