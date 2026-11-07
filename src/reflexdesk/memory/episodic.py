"""L2 episodic memory: журнал прогнозов, решений, исходов и реестр прогонов (append-only)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

from reflexdesk.config import get_settings
from reflexdesk.schemas import Prediction, PredictionStatus, ensure_utc, utcnow

metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("prediction_id", String(64), primary_key=True),
    Column("run_id", String(64), index=True, nullable=False),
    Column("ticker", String(16), index=True, nullable=False),
    Column("direction", String(8), nullable=False),
    Column("horizon_days", Integer, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("thesis", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("check_at", DateTime(timezone=True), index=True, nullable=False),
    Column("status", String(16), index=True, nullable=False),
    Column("actual_return", Float),
    Column("verified_at", DateTime(timezone=True)),
)

decisions = Table(
    "decisions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), index=True, nullable=False),
    Column("node", String(64), nullable=False),
    Column("artifact_json", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

outcomes = Table(
    "outcomes", metadata,
    Column("run_id", String(64), primary_key=True),
    Column("report_path", Text),
    Column("validation_json", Text),
    Column("cost_usd", Float),
    Column("latency_s", Float),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

runs = Table(
    "runs", metadata,
    Column("run_id", String(64), primary_key=True),
    Column("query", Text, nullable=False),
    Column("as_of_date", String(10)),
    Column("tags", String(128)),
    Column("requester", String(64), index=True),
    Column("status", String(16), nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

STALE_AFTER = timedelta(days=2)
EVAL_TAG_PREFIX = "eval:"


def _dump(artifact: Any) -> str:
    if hasattr(artifact, "model_dump"):
        artifact = artifact.model_dump(mode="json")
    return json.dumps(artifact, ensure_ascii=False, default=str)


def _eval_run_ids():
    """Подзапрос run_id прогонов evals: они не входят в продуктовые метрики."""
    return select(runs.c.run_id).where(runs.c.tags.like(f"{EVAL_TAG_PREFIX}%"))


def _is_working_prediction():
    return predictions.c.run_id.not_in(_eval_run_ids())


class EpisodicStore:
    """Доступ к L2; записи только добавляются, у прогнозов меняется лишь статус."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        metadata.create_all(engine)
        self._migrate()

    def _migrate(self) -> None:
        """Идемпотентное добавление колонок, появившихся после создания таблиц."""
        existing = {c["name"] for c in inspect(self._engine).get_columns("runs")}
        if "requester" not in existing:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE runs ADD COLUMN requester VARCHAR(64)"))

    def start_run(self, run_id: str, query: str, as_of_date: str | None, tags: str = "", requester: str | None = None) -> None:
        now = utcnow()
        with self._engine.begin() as conn:
            conn.execute(insert(runs).values(
                run_id=run_id, query=query, as_of_date=as_of_date, tags=tags, requester=requester,
                status="running", created_at=now, updated_at=now,
            ))

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._engine.begin() as conn:
            conn.execute(update(runs).where(runs.c.run_id == run_id).values(status=status, error=error, updated_at=utcnow()))

    def run(self, run_id: str) -> dict | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.run_id == run_id)).mappings().first()
        return dict(row) if row else None

    def runs_by_status(self, status: str) -> list[dict]:
        with self._engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(runs).where(runs.c.status == status)).mappings()]

    def runs_of(self, requester: str, status: str | None = None, limit: int = 20) -> list[dict]:
        """Прогоны пользователя, от новых к старым."""
        query = select(runs).where(runs.c.requester == requester).order_by(runs.c.created_at.desc()).limit(limit)
        if status:
            query = query.where(runs.c.status == status)
        with self._engine.connect() as conn:
            return [dict(r) for r in conn.execute(query).mappings()]

    def runs_older_than(self, cutoff: datetime) -> list[str]:
        with self._engine.connect() as conn:
            return list(conn.execute(select(runs.c.run_id).where(runs.c.created_at < cutoff)).scalars())

    def record_decision(self, run_id: str, node: str, artifact: Any) -> None:
        with self._engine.begin() as conn:
            conn.execute(insert(decisions).values(run_id=run_id, node=node, artifact_json=_dump(artifact), created_at=utcnow()))

    def decisions(self, run_id: str, node: str | None = None) -> list[dict]:
        query = select(decisions).where(decisions.c.run_id == run_id).order_by(decisions.c.id)
        if node:
            query = query.where(decisions.c.node == node)
        with self._engine.connect() as conn:
            return [{**r, "artifact": json.loads(r["artifact_json"])} for r in conn.execute(query).mappings()]

    def record_outcome(self, run_id: str, report_path: str | None, validation: Any, cost_usd: float, latency_s: float) -> None:
        with self._engine.begin() as conn:
            conn.execute(insert(outcomes).values(
                run_id=run_id, report_path=report_path, validation_json=_dump(validation),
                cost_usd=cost_usd, latency_s=latency_s, created_at=utcnow(),
            ))

    def outcome(self, run_id: str) -> dict | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(outcomes).where(outcomes.c.run_id == run_id)).mappings().first()
        return dict(row) if row else None

    def outcomes(self, run_ids: list[str] | None = None, working_only: bool = False) -> list[dict]:
        query = select(outcomes)
        if working_only:
            query = query.where(outcomes.c.run_id.not_in(_eval_run_ids()))
        if run_ids is not None:
            query = query.where(outcomes.c.run_id.in_(run_ids))
        with self._engine.connect() as conn:
            return [dict(r) for r in conn.execute(query).mappings()]

    def add_predictions(self, items: list[Prediction]) -> None:
        if not items:
            return
        with self._engine.begin() as conn:
            conn.execute(insert(predictions), [p.model_dump() for p in items])

    def predictions(
        self,
        ticker: str | None = None,
        status: str | None = None,
        run_id: str | None = None,
        requester: str | None = None,
        verified_since: datetime | None = None,
        working_only: bool = False,
    ) -> list[Prediction]:
        query = select(predictions).order_by(predictions.c.created_at)
        if working_only:
            query = query.where(_is_working_prediction())
        if requester:
            query = query.where(predictions.c.run_id.in_(select(runs.c.run_id).where(runs.c.requester == requester)))
        if verified_since:
            query = query.where(predictions.c.verified_at > verified_since)
        if ticker:
            query = query.where(predictions.c.ticker == ticker.upper())
        if status:
            query = query.where(predictions.c.status == status)
        if run_id:
            query = query.where(predictions.c.run_id == run_id)
        with self._engine.connect() as conn:
            return [self._to_prediction(r) for r in conn.execute(query).mappings()]

    def due_predictions(self, now: datetime | None = None) -> list[Prediction]:
        """Созревшие pending-прогнозы рабочих прогонов (check_at ≤ now)."""
        now = now or utcnow()
        query = select(predictions).where(predictions.c.status == "pending", predictions.c.check_at <= now, _is_working_prediction())
        with self._engine.connect() as conn:
            return [self._to_prediction(r) for r in conn.execute(query).mappings()]

    def set_status(self, prediction_id: str, status: PredictionStatus, actual_return: float | None) -> None:
        """Единственная разрешённая мутация: pending → hit/miss/expired."""
        with self._engine.begin() as conn:
            changed = conn.execute(
                update(predictions)
                .where(predictions.c.prediction_id == prediction_id, predictions.c.status == "pending")
                .values(status=status, actual_return=actual_return, verified_at=utcnow())
            ).rowcount
        if not changed:
            raise ValueError(f"prediction {prediction_id} is not pending")

    def counts(self, now: datetime | None = None) -> dict[str, int]:
        """pending и просроченные >2 дней pending-прогнозы рабочих прогонов."""
        now = now or utcnow()
        pending_filter = (predictions.c.status == "pending", _is_working_prediction())
        with self._engine.connect() as conn:
            pending = conn.execute(select(func.count()).where(*pending_filter)).scalar_one()
            stale = conn.execute(
                select(func.count()).where(*pending_filter, predictions.c.check_at < now - STALE_AFTER)
            ).scalar_one()
        return {"pending": pending, "stale": stale}

    @staticmethod
    def _to_prediction(row) -> Prediction:
        data = dict(row)
        for key in ("created_at", "check_at", "verified_at"):
            data[key] = ensure_utc(data[key])
        return Prediction.model_validate(data)


@lru_cache(maxsize=1)
def get_episodic() -> EpisodicStore:
    """Синглтон L2 по DATABASE_URL (Postgres) или локальному SQLite."""
    url = get_settings().resolved_database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return EpisodicStore(create_engine(url, pool_pre_ping=True, future=True))
