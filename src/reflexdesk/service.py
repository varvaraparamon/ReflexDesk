"""Сервис прогонов: старт, возобновление с чекпоинта, статус и отчёт."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

from reflexdesk.agents.services import Services, default_services
from reflexdesk.graph import FULL, GraphConfig, build_graph
from reflexdesk.memory.working import open_checkpointer
from reflexdesk.obs import metrics, tracing
from reflexdesk.obs.context import bind
from reflexdesk.obs.logging import get_logger
from reflexdesk.state import DeskState, initial_state
from reflexdesk.tools.filings import release_run

log = get_logger("service")

RECURSION_LIMIT = 60
MAX_RESUME_ATTEMPTS = 10


class DeskService:
    """Фасад запуска графа поверх чекпоинтера L1 и журнала L2."""

    def __init__(self, services: Services | None = None, config: GraphConfig = FULL) -> None:
        self.services = services or default_services()
        self.config = config
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = {}

    @cached_property
    def checkpointer(self):
        return open_checkpointer()

    @cached_property
    def graph(self):
        return build_graph(self.services, self.checkpointer, self.config)

    def new_run(self, query: str, as_of: date | None = None, tags: str = "", requester: str | None = None) -> str:
        """Регистрирует прогон в L2 и возвращает run_id."""
        run_id = uuid.uuid4().hex
        self.services.episodic.start_run(
            run_id, query, as_of.isoformat() if as_of else None, tags=tags or self.config.name, requester=requester,
        )
        return run_id

    def execute(self, run_id: str) -> DeskState:
        """Исполняет прогон; если есть чекпоинт — продолжает с него (FR-10). Один прогон — не более одного исполнения."""
        with self._lock:
            if run_id in self._active:
                raise RuntimeError(f"run {run_id} is already executing")
            self._active.add(run_id)
        try:
            return self._execute(run_id)
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _execute(self, run_id: str) -> DeskState:
        run = self.services.episodic.run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] != "running":
            self.services.episodic.set_run_status(run_id, "running")
        cfg = self._thread(run_id)
        has_checkpoint = self.graph.get_state(cfg).values != {}
        payload = None if has_checkpoint else initial_state(run_id, run["query"], run["as_of_date"])
        started = time.perf_counter()
        scenario = "unknown"
        tags = [t for t in (run["tags"] or "").split(",") if t]
        with (
            bind(run_id=run_id),
            tracing.run_scope(run_id, tags, {"graph_config": self.config.name}),
            tracing.observe("run", "chain", input={"query": run["query"], "as_of": run["as_of_date"]}) as span,
        ):
            try:
                final: DeskState = self.graph.invoke(payload, cfg)
            except Exception as exc:
                self.services.episodic.set_run_status(run_id, "interrupted", error=f"{type(exc).__name__}: {exc}")
                metrics.graph_runs_total.labels(scenario, "error").inc()
                log.exception("run interrupted", run_id=run_id)
                raise
            finally:
                release_run(run_id)
                self.services.router.ledger.release(run_id)
            scenario = final.get("scenario") or scenario
            status = final.get("status", "done")
            span.update(output={"status": status, "cost_usd": final.get("cost_usd"), "report": final.get("report_path")})
        self.services.episodic.set_run_status(run_id, status)
        metrics.graph_runs_total.labels(scenario, status).inc()
        metrics.graph_run_duration_seconds.observe(time.perf_counter() - started)
        self._score(run_id, final)
        tracing.flush()
        return final

    def analyze(self, query: str, as_of: date | None = None, tags: str = "") -> DeskState:
        """Синхронный прогон от запроса до отчёта."""
        return self.execute(self.new_run(query, as_of, tags))

    def resume(self, statuses: tuple[str, ...] = ("interrupted",)) -> list[str]:
        """Продолжает прогоны в указанных статусах с чекпоинта; после MAX_RESUME_ATTEMPTS неудач — failed."""
        resumed: list[str] = []
        for status in statuses:
            for run in self.services.episodic.runs_by_status(status):
                run_id = run["run_id"]
                if run_id in self._active:
                    continue
                try:
                    self.execute(run_id)
                    resumed.append(run_id)
                    self._attempts.pop(run_id, None)
                except Exception as exc:
                    self._attempts[run_id] = self._attempts.get(run_id, 0) + 1
                    log.warning("resume failed", run_id=run_id, attempt=self._attempts[run_id], error=str(exc))
                    if self._attempts[run_id] >= MAX_RESUME_ATTEMPTS:
                        self.services.episodic.set_run_status(
                            run_id, "failed", error=f"resume attempts exhausted: {type(exc).__name__}: {exc}",
                        )
                        self._attempts.pop(run_id, None)
        return resumed

    def resume_forever(self, interval_s: float) -> None:
        """Фоновый цикл: осиротевшие running при старте, затем периодически interrupted."""
        self.resume(("running", "interrupted"))
        while True:
            time.sleep(interval_s)
            self.resume()

    def state(self, run_id: str) -> DeskState:
        return self.graph.get_state(self._thread(run_id)).values

    def summary(self, run_id: str) -> dict[str, Any] | None:
        """Статус прогона и краткая выжимка DeskState."""
        run = self.services.episodic.run(run_id)
        if run is None:
            return None
        values = self.state(run_id)
        validation = values.get("validation")
        return {
            "run_id": run_id,
            "status": run["status"],
            "requester": run["requester"],
            "created_at": run["created_at"],
            "query": run["query"],
            "as_of_date": run["as_of_date"],
            "scenario": values.get("scenario"),
            "ticker": values.get("ticker"),
            "evidence_count": len(values.get("evidence", [])),
            "validation": validation.model_dump() if validation else None,
            "validation_round": values.get("validation_round", 0),
            "risk_score": values.get("risk_score"),
            "lessons_used": values.get("lessons_used", []),
            "cost_usd": round(values.get("cost_usd", 0.0), 6),
            "flags": values.get("flags", []),
            "errors": values.get("errors", []),
            "report_path": values.get("report_path"),
            "pdf_path": values.get("pdf_path"),
        }

    def report(self, run_id: str) -> tuple[str, str | None] | None:
        """(markdown, pdf_path) готового отчёта."""
        values = self.state(run_id)
        path = values.get("report_path")
        if not path:
            return None
        return Path(path).read_text(encoding="utf-8"), values.get("pdf_path")

    @staticmethod
    def _thread(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}

    @staticmethod
    def _score(run_id: str, final: DeskState) -> None:
        validation = final.get("validation")
        if validation is not None:
            tracing.score_run(run_id, "validator_verdict", 1.0 if validation.verdict == "ok" else 0.0)
        tracing.score_run(run_id, "cost_usd", float(final.get("cost_usd", 0.0)))
