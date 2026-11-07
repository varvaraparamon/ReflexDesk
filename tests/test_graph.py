"""Интеграция графа: полный путь, раунды валидации, reject, возобновление после сбоя (AC-4), Reflector (AC-3)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeGates, FakeRouter, memory_knowledge

from reflexdesk import tools
from reflexdesk.agents.analyst import AnalystNode
from reflexdesk.agents.reflector import Reflector
from reflexdesk.agents.services import Services
from reflexdesk.config import get_settings
from reflexdesk.graph import SINGLE_ANALYST, GraphConfig
from reflexdesk.prompting import get_library
from reflexdesk.schemas import Evidence, LessonContent
from reflexdesk.service import DeskService
from reflexdesk.tools.base import ToolResult


@pytest.fixture(autouse=True)
def fake_market_tool(monkeypatch):
    def market(ticker: str, start=None, end=None, interval="1d") -> ToolResult:
        return ToolResult(tool="get_market_data", evidence=[
            Evidence(fact="Цена закрытия TEST", value=100.5, source_url="https://example.com/test", tool="get_market_data"),
        ])

    monkeypatch.setitem(tools.REGISTRY, "get_market_data", market)


def make_service(episodic, router=None, gates=None, config: GraphConfig | None = None, knowledge=None) -> DeskService:
    services = Services(
        settings=get_settings(), router=router or FakeRouter(), gates=gates or FakeGates(),
        library=get_library(), episodic=episodic, knowledge_factory=lambda: knowledge or memory_knowledge(),
    )
    return DeskService(services, config=config) if config else DeskService(services)


def test_full_run_writes_report_and_predictions(episodic):
    service = make_service(episodic)
    final = service.analyze("Проанализируй TEST")
    assert final["status"] == "done" and final["validation"].verdict == "ok"
    predictions = episodic.predictions(run_id=final["run_id"])
    assert len(predictions) == 1 and predictions[0].horizon_days == 30
    assert episodic.outcome(final["run_id"]) is not None
    nodes = [d["node"] for d in episodic.decisions(final["run_id"])]
    assert nodes[:4] == ["intake", "orchestrator", "collector", "analyst"] and "validator" in nodes
    assert {"bull", "bear"} <= set(nodes) and "risk_officer" not in nodes


def test_risky_draft_routes_through_risk_officer(episodic):
    final = make_service(episodic, gates=FakeGates(risk=9)).analyze("Проанализируй TEST")
    assert "risk_officer" in [d["node"] for d in episodic.decisions(final["run_id"])]


def test_validation_exhausted_marks_unvalidated(episodic):
    router = FakeRouter(validator_verdicts=["fail"])
    final = make_service(episodic, router=router).analyze("Проанализируй TEST")
    assert final["validation_round"] == 2 and "UNVALIDATED" in final["flags"]
    assert router.calls["chat"] == 2
    assert "UNVALIDATED" in Path(final["report_path"]).read_text(encoding="utf-8")


def test_second_round_passes(episodic):
    router = FakeRouter(validator_verdicts=["fail", "ok"])
    final = make_service(episodic, router=router).analyze("Проанализируй TEST")
    assert final["validation"].verdict == "ok" and final["validation_round"] == 2 and "UNVALIDATED" not in final["flags"]


def test_reject_scenario(episodic):
    final = make_service(episodic, gates=FakeGates(scenario="reject")).analyze("Напиши стих")
    assert final["status"] == "rejected" and not episodic.predictions(run_id=final["run_id"])


def test_single_analyst_ablation_config(episodic):
    final = make_service(episodic, config=SINGLE_ANALYST).analyze("Проанализируй TEST")
    nodes = {d["node"] for d in episodic.decisions(final["run_id"])}
    assert final["status"] == "done" and not nodes & {"bull", "bear", "validator"}


def test_resume_after_crash_continues_from_checkpoint(episodic, monkeypatch):
    router = FakeRouter()
    service = make_service(episodic, router=router)
    original = AnalystNode.__call__
    state = {"crashed": False}

    def crashing(self, desk_state):
        if not state["crashed"]:
            state["crashed"] = True
            raise RuntimeError("simulated kill -9")
        return original(self, desk_state)

    monkeypatch.setattr(AnalystNode, "__call__", crashing)
    run_id = service.new_run("Проанализируй TEST", date(2024, 3, 1))
    with pytest.raises(RuntimeError):
        service.execute(run_id)
    assert episodic.run(run_id)["status"] == "interrupted"
    assert router.calls["Plan"] == 1

    resumed = make_service(episodic, router=router)
    assert resumed.resume() == [run_id]
    assert router.calls["Plan"] == 1
    assert episodic.run(run_id)["status"] == "done"
    assert [d["node"] for d in episodic.decisions(run_id)].count("collector") == 1


def test_reflector_verifies_forced_predictions_and_writes_lessons(episodic, semantic_store, monkeypatch):
    from reflexdesk.memory import Knowledge
    from reflexdesk.memory.knowledge import DocumentMemory, LessonMemory, ProfileMemory

    knowledge = Knowledge(DocumentMemory(semantic_store), ProfileMemory(semantic_store), LessonMemory(semantic_store))
    final = make_service(episodic, knowledge=knowledge).analyze("Проанализируй TEST", date(2024, 3, 1))
    prediction = episodic.predictions(run_id=final["run_id"])[0]
    assert prediction.check_at == datetime(2024, 3, 31, 23, 59, 59, tzinfo=UTC)

    monkeypatch.setattr("reflexdesk.agents.reflector.total_return", lambda ticker, start, end: -0.08)
    lesson = LessonContent(ticker="TEST", error_type="wrong_direction", context="c", expectation="up 0.6",
                           outcome="-8%", cause="перегрев", rule="при перегреве снижать уверенность")
    monkeypatch.setattr(Reflector, "_extract", lambda self, messages, existing, report: [SimpleNamespace(id="new", content=lesson)])

    services = Services(settings=get_settings(), router=FakeRouter(), gates=FakeGates(), library=get_library(),
                        episodic=episodic, knowledge_factory=lambda: knowledge)
    report = Reflector(services).run(now=prediction.check_at + timedelta(days=1))
    assert report.verified == {"miss": 1} and report.lessons_added == 1
    assert episodic.predictions(run_id=final["run_id"])[0].status == "miss"
    assert knowledge.lessons.active("TEST")[0]["lesson"]["rule"] == lesson.rule


def test_degraded_validation_never_accepted(episodic, monkeypatch):
    from reflexdesk.agents import validator as validator_module

    original = validator_module.ValidatorNode.__call__

    def degraded(self, state):
        update = original(self, state)
        update["validation"].degraded = True
        return update

    monkeypatch.setattr(validator_module.ValidatorNode, "__call__", degraded)
    final = make_service(episodic).analyze("Проанализируй TEST")
    assert final["validation"].verdict == "ok" and final["validation"].degraded
    assert final["validation_round"] == 2 and "UNVALIDATED" in final["flags"]


def test_resume_gives_up_after_attempts(episodic, monkeypatch):
    from reflexdesk import service as service_module

    monkeypatch.setattr(service_module, "MAX_RESUME_ATTEMPTS", 2)
    service = make_service(episodic)
    monkeypatch.setattr(AnalystNode, "__call__", lambda self, state: (_ for _ in ()).throw(RuntimeError("jev down")))
    run_id = service.new_run("Проанализируй TEST")
    with pytest.raises(RuntimeError):
        service.execute(run_id)
    assert service.resume() == [] and episodic.run(run_id)["status"] == "interrupted"
    assert service.resume() == [] and episodic.run(run_id)["status"] == "failed"
    assert "resume attempts exhausted" in episodic.run(run_id)["error"]
