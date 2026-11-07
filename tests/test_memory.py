"""L2 (иммутабельность, созревание) и L3 (look-ahead фильтр, свежесть, дедуп, supersede)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from reflexdesk.memory.knowledge import (
    DocumentMemory,
    LessonMemory,
    ProfileMemory,
    chunk_text,
)
from reflexdesk.memory.semantic import DOCUMENTS, MemoryRecord, freshness
from reflexdesk.schemas import Lesson, Prediction

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _prediction(run_id: str = "r1", horizon: int = 30, created: datetime = T0) -> Prediction:
    return Prediction(run_id=run_id, ticker="AAPL", direction="up", horizon_days=horizon, confidence=0.6, thesis="t", created_at=created)


def test_prediction_check_at_derived():
    assert _prediction(horizon=90).check_at == T0 + timedelta(days=90)


def test_due_and_status_transition(episodic):
    early, late = _prediction(horizon=30), _prediction(horizon=180)
    episodic.add_predictions([early, late])
    due = episodic.due_predictions(T0 + timedelta(days=31))
    assert [p.prediction_id for p in due] == [early.prediction_id]
    episodic.set_status(early.prediction_id, "hit", 0.05)
    with pytest.raises(ValueError):
        episodic.set_status(early.prediction_id, "miss", -0.1)
    assert episodic.predictions(status="hit")[0].actual_return == 0.05


def test_eval_runs_excluded_from_working_metrics(episodic):
    episodic.start_run("live", "q", None, tags="full")
    episodic.start_run("golden", "q", None, tags="eval:golden:full")
    live, golden = _prediction(run_id="live"), _prediction(run_id="golden")
    episodic.add_predictions([live, golden])
    later = T0 + timedelta(days=31)
    assert [p.run_id for p in episodic.due_predictions(later)] == ["live"]
    assert [p.run_id for p in episodic.predictions(working_only=True)] == ["live"]
    assert len(episodic.predictions()) == 2
    assert episodic.counts(later)["pending"] == 1


def test_counts_pending_and_stale(episodic):
    episodic.add_predictions([_prediction(horizon=30)])
    counts = episodic.counts(T0 + timedelta(days=40))
    assert counts == {"pending": 1, "stale": 1}


def test_decisions_trajectory(episodic):
    episodic.record_decision("r1", "intake", {"scenario": "analyze_ticker"})
    episodic.record_decision("r1", "analyst", {"analysis_draft": {"theses": ["a"]}})
    assert [d["node"] for d in episodic.decisions("r1")] == ["intake", "analyst"]


def test_freshness_decay():
    assert freshness(1.0, 35, 0.02) == pytest.approx(0.4966, abs=1e-3)
    assert freshness(1.0, 0, 0.02) == 1.0


def test_chunking_overlap():
    chunks = chunk_text("абзац\n" * 600, size=500, overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 500 for c in chunks)


def test_lookahead_filter(semantic_store):
    docs = DocumentMemory(semantic_store)
    docs.ingest("Apple dividends raised", url="u1", title="t", ticker="AAPL", published_at=T0, kind="news")
    docs.ingest("Apple dividends cut in future", url="u2", title="t", ticker="AAPL", published_at=T0 + timedelta(days=60), kind="news")
    hits = docs.search("Apple dividends", as_of=T0 + timedelta(days=1), ticker="AAPL")
    assert [h.payload["url"] for h in hits] == ["u1"]


def test_fresher_document_ranks_higher(semantic_store):
    docs = DocumentMemory(semantic_store)
    docs.ingest("Apple revenue growth services", url="old", title="t", ticker="AAPL", published_at=T0 - timedelta(days=400), kind="news")
    docs.ingest("Apple revenue growth iphone", url="new", title="t", ticker="AAPL", published_at=T0 - timedelta(days=2), kind="news")
    hits = docs.search("Apple revenue growth", as_of=T0, ticker="AAPL")
    assert hits[0].payload["url"] == "new"


def test_dedup_merges_near_duplicates(semantic_store):
    record = MemoryRecord(text="identical text about margins", ticker="AAPL", published_at=T0)
    first = semantic_store.upsert(DOCUMENTS, [record])
    second = semantic_store.upsert(DOCUMENTS, [MemoryRecord(text="identical text about margins", ticker="AAPL", published_at=T0)])
    assert first == second


def test_lessons_supersede_and_relevance(semantic_store):
    lessons = LessonMemory(semantic_store)
    old = Lesson(ticker="AAPL", error_type="overconfidence", context="c", expectation="up 0.9", outcome="miss",
                 cause="тарифы", rule="снижать уверенность при тарифном риске", created_at=T0)
    new = Lesson(ticker="AAPL", error_type="ignored_risk", context="c", expectation="up 0.8", outcome="miss",
                 cause="тарифы", rule="учитывать тарифный риск AAPL", created_at=T0)
    lessons.add([old, new])
    lessons.supersede(old.lesson_id, new.lesson_id)
    active_ids = {row["id"] for row in lessons.active("AAPL")}
    assert active_ids == {new.lesson_id}
    hits = lessons.relevant("AAPL тарифный риск уверенность", as_of=T0 + timedelta(days=1), ticker="AAPL", sector=None)
    assert [h.id for h in hits] == [new.lesson_id]


def test_profile_versioning(semantic_store):
    profiles = ProfileMemory(semantic_store)
    assert profiles.update("MSFT", sector="Technology", metrics={"P/E": "35"}, stance="up", as_of=T0) == 1
    assert profiles.update("MSFT", sector=None, metrics={"EV/EBITDA": "24"}, stance="flat", as_of=T0) == 2
    profile = profiles.get("MSFT")
    assert profile["metrics"] == {"P/E": "35", "EV/EBITDA": "24"} and profile["sector"] == "Technology"
