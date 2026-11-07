"""HTTP API: асинхронный запуск, статус, отчёт, журнал прогнозов, метрики, health."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_graph import make_service

from reflexdesk import api


@pytest.fixture
def client(episodic, monkeypatch):
    service = make_service(episodic)
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "get_gateway", lambda: service.services.gates)
    with TestClient(api.app) as test_client:
        yield test_client


def test_analyze_flow(client):
    run_id = client.post("/v1/analyze", json={"query": "Проанализируй TEST", "as_of_date": "2024-03-01"}).json()["run_id"]
    status = client.get(f"/v1/runs/{run_id}").json()
    assert status["status"] == "done" and status["ticker"] == "TEST" and status["as_of_date"] == "2024-03-01"
    report = client.get(f"/v1/runs/{run_id}/report").json()
    assert "## Прогнозы" in report["markdown"]
    predictions = client.get("/v1/predictions", params={"ticker": "TEST"}).json()
    assert len(predictions) == 1 and predictions[0]["status"] == "pending"
    summary = client.get("/v1/metrics/summary").json()
    assert summary["reports"]["count"] == 1 and summary["predictions"]["total"] == 1


def test_unknown_run_404(client):
    assert client.get("/v1/runs/nope").status_code == 404


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    assert "llm_tokens_total" in client.get("/metrics").text


def test_readyz_fails_when_jev_down(client, monkeypatch):
    class Down:
        def decide(self, spec, gate=""):
            raise RuntimeError("jev down")

    monkeypatch.setattr(api, "get_gateway", lambda: Down())
    response = client.get("/readyz")
    assert response.status_code == 503 and response.json()["detail"]["jev"].startswith("error")


def test_runs_by_requester_and_single_active_run(client):
    first = client.post("/v1/analyze", json={"query": "Проанализируй TEST", "requester": "tg:1"})
    assert first.status_code == 202
    runs = client.get("/v1/runs", params={"requester": "tg:1"}).json()
    assert [r["run_id"] for r in runs] == [first.json()["run_id"]] and runs[0]["status"] == "done"
    predictions = client.get("/v1/predictions", params={"requester": "tg:1"}).json()
    assert len(predictions) == 1
    assert client.get("/v1/runs", params={"requester": "tg:2"}).json() == []


def test_busy_requester_rejected(client, episodic):
    episodic.start_run("busy-run", "q", None, requester="tg:9")
    response = client.post("/v1/analyze", json={"query": "Проанализируй TEST", "requester": "tg:9"})
    assert response.status_code == 409
    episodic.set_run_status("busy-run", "interrupted")
    assert client.post("/v1/analyze", json={"query": "x", "requester": "tg:9"}).status_code == 409
