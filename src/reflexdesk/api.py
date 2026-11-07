"""HTTP API деска (FastAPI): прогоны, отчёты, журнал прогнозов, метрики, health."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from reflexdesk.evals.summary import metrics_summary
from reflexdesk.gateway import get_gateway
from reflexdesk.gateway.gates import GateSpecs
from reflexdesk.obs.logging import get_logger
from reflexdesk.service import DeskService

log = get_logger("api")

RESUME_INTERVAL_S = 60.0
ACTIVE_STATUSES = ("running", "interrupted")


class AnalyzeRequest(BaseModel):
    query: str
    as_of_date: date | None = None
    requester: str | None = None


class AnalyzeResponse(BaseModel):
    run_id: str


@lru_cache(maxsize=1)
def get_service() -> DeskService:
    return DeskService()


def _run_safely(run_id: str) -> None:
    try:
        get_service().execute(run_id)
    except Exception:
        log.exception("background run failed", run_id=run_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(
        target=get_service().resume_forever, kwargs={"interval_s": RESUME_INTERVAL_S},
        name="resume-loop", daemon=True,
    ).start()
    yield


app = FastAPI(title="ReflexDesk", version="0.1.0", lifespan=lifespan)


@app.post("/v1/analyze", response_model=AnalyzeResponse, status_code=202)
def analyze(request: AnalyzeRequest, background: BackgroundTasks) -> AnalyzeResponse:
    service = get_service()
    if request.requester and any(
        service.services.episodic.runs_of(request.requester, status=s, limit=1) for s in ACTIVE_STATUSES
    ):
        raise HTTPException(409, "requester already has an active analysis")
    run_id = service.new_run(request.query, request.as_of_date, requester=request.requester)
    background.add_task(_run_safely, run_id)
    return AnalyzeResponse(run_id=run_id)


@app.get("/v1/runs")
def runs(requester: str, status: str | None = None, limit: int = 20) -> list[dict]:
    """Прогоны пользователя, от новых к старым."""
    rows = get_service().services.episodic.runs_of(requester, status=status, limit=limit)
    keys = ("run_id", "status", "query", "as_of_date", "created_at", "updated_at", "error")
    return [{k: r[k] for k in keys} for r in rows]


@app.get("/v1/runs/{run_id}")
def run_status(run_id: str) -> dict:
    summary = get_service().summary(run_id)
    if summary is None:
        raise HTTPException(404, "run not found")
    return summary


@app.get("/v1/runs/{run_id}/report")
def run_report(run_id: str) -> dict:
    report = get_service().report(run_id)
    if report is None:
        raise HTTPException(404, "report not ready")
    markdown, pdf_path = report
    return {"markdown": markdown, "pdf_url": f"/v1/runs/{run_id}/report.pdf" if pdf_path else None}


@app.get("/v1/runs/{run_id}/report.pdf")
def run_report_pdf(run_id: str) -> FileResponse:
    report = get_service().report(run_id)
    if report is None or report[1] is None or not Path(report[1]).exists():
        raise HTTPException(404, "pdf not available")
    return FileResponse(report[1], media_type="application/pdf", filename=f"{run_id}.pdf")


@app.get("/v1/predictions")
def predictions(
    ticker: str | None = None,
    status: str | None = None,
    run_id: str | None = None,
    requester: str | None = None,
    verified_since: datetime | None = None,
) -> list[dict]:
    store = get_service().services.episodic
    found = store.predictions(ticker=ticker, status=status, run_id=run_id, requester=requester, verified_since=verified_since)
    return [p.model_dump(mode="json") for p in found]


@app.get("/v1/metrics/summary")
def summary() -> dict:
    return metrics_summary(get_service().services.episodic)


@app.get("/metrics")
def prometheus() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


def _probe(check) -> str:
    try:
        check()
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
    return "ok"


@app.get("/readyz")
def readyz() -> dict:
    """Готовность зависимостей: L1, L2, L3 и Jev-модель гейтов."""
    service = get_service()
    gateway = GateSpecs.needs_validation("Выручка 10 млрд USD")
    checks = {
        "l1": _probe(lambda: service.checkpointer),
        "l2": _probe(service.services.episodic.counts),
        "l3": _probe(lambda: service.services.knowledge.profiles.get("READYZ")),
        "jev": _probe(lambda: get_gateway().decide(gateway, gate="readyz")),
    }
    if any(v != "ok" for v in checks.values()):
        raise HTTPException(503, checks)
    return checks
