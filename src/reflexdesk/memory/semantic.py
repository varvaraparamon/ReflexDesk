"""L3 semantic memory в Qdrant: hybrid (dense ∥ BM25 -> RRF) -> rerank -> свежесть -> look-ahead фильтр; дедуп при записи."""

from __future__ import annotations

import atexit
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from reflexdesk.config import get_settings
from reflexdesk.memory.embeddings import Embedder, get_embedder
from reflexdesk.schemas import utcnow

DENSE = "dense"
SPARSE = "bm25"
PREFETCH_LIMIT = 50
FUSED_LIMIT = 20
DEDUP_THRESHOLD = 0.97
DAY_S = 86400.0


@dataclass(frozen=True)
class CollectionSpec:
    """Параметры коллекции: λ свежести (1/день)."""

    name: str
    decay: float


DOCUMENTS = CollectionSpec("documents", 0.02)
PROFILES = CollectionSpec("ticker_profiles", 0.0)
LESSONS = CollectionSpec("lessons", 0.002)
COLLECTIONS = (DOCUMENTS, PROFILES, LESSONS)


@dataclass
class MemoryRecord:
    """Запись L3: текст, тикер, дата публикации, метаданные."""

    text: str
    ticker: str | None = None
    sector: str | None = None
    published_at: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class Hit:
    """Результат поиска с итоговым скором."""

    id: str
    text: str
    payload: dict[str, Any]
    score: float


def freshness(score: float, age_days: float, decay: float) -> float:
    """score × exp(−λ·age)."""
    return score * math.exp(-decay * max(age_days, 0.0))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def profile_id(ticker: str) -> str:
    """Детерминированный id профиля тикера."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reflexdesk:profile:{ticker.upper()}"))


class SemanticStore:
    """Операции над коллекциями L3."""

    def __init__(self, client: QdrantClient, embedder: Embedder) -> None:
        self._client = client
        self._embedder = embedder
        self.ensure_collections()

    def ensure_collections(self) -> None:
        for spec in COLLECTIONS:
            if self._client.collection_exists(spec.name):
                continue
            self._client.create_collection(
                spec.name,
                vectors_config={DENSE: models.VectorParams(size=self._embedder.dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )

    def upsert(self, spec: CollectionSpec, records: list[MemoryRecord]) -> list[str]:
        """Запись с дедупликацией: косинус > 0.97 к существующей -> merge и обновление last_seen."""
        if not records:
            return []
        texts = [r.text for r in records]
        dense, sparse = self._embedder.dense(texts), self._embedder.sparse(texts)
        now = utcnow()
        ids, points = [], []
        for record, dv, sv in zip(records, dense, sparse, strict=True):
            duplicate = None if record.id else self._nearest_duplicate(spec, dv)
            if duplicate is not None:
                self._client.set_payload(spec.name, payload={"last_seen": now.isoformat(), **record.meta}, points=[duplicate])
                ids.append(duplicate)
                continue
            point_id = record.id or str(uuid.uuid4())
            published = record.published_at or now
            payload = {
                "text": record.text,
                "ticker": record.ticker.upper() if record.ticker else None,
                "sector": record.sector,
                "published_at": published.isoformat(),
                "published_ts": published.timestamp(),
                "last_seen": now.isoformat(),
                **record.meta,
            }
            points.append(models.PointStruct(id=point_id, vector={DENSE: dv, SPARSE: sv}, payload=payload))
            ids.append(point_id)
        if points:
            self._client.upsert(spec.name, points=points)
        return ids

    def search(
        self,
        spec: CollectionSpec,
        query: str,
        *,
        as_of: datetime,
        ticker: str | None = None,
        sector: str | None = None,
        top_k: int = 8,
        extra_filter: list[models.Condition] | None = None,
    ) -> list[Hit]:
        """Hybrid retrieval -> rerank -> вес свежести; только published_at ≤ as_of."""
        conditions: list[models.Condition] = [
            models.FieldCondition(key="published_ts", range=models.Range(lte=as_of.timestamp())),
            *(extra_filter or []),
        ]
        should: list[models.Condition] = []
        if ticker:
            should.append(models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper())))
        if sector:
            should.append(models.FieldCondition(key="sector", match=models.MatchValue(value=sector)))
        flt = models.Filter(must=conditions, should=should or None)
        fused = self._client.query_points(
            spec.name,
            prefetch=[
                models.Prefetch(query=self._embedder.dense([query])[0], using=DENSE, limit=PREFETCH_LIMIT, filter=flt),
                models.Prefetch(query=self._embedder.sparse_query(query), using=SPARSE, limit=PREFETCH_LIMIT, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=FUSED_LIMIT,
            with_payload=True,
        ).points
        if not fused:
            return []
        rerank = self._embedder.rerank(query, [p.payload["text"] for p in fused])
        ranked = sorted(zip(fused, rerank, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
        hits = []
        for point, logit in ranked:
            age = (as_of.timestamp() - point.payload["published_ts"]) / DAY_S
            hits.append(Hit(str(point.id), point.payload["text"], point.payload, freshness(_sigmoid(logit), age, spec.decay)))
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def get(self, spec: CollectionSpec, point_id: str) -> dict[str, Any] | None:
        found = self._client.retrieve(spec.name, ids=[point_id], with_payload=True)
        return found[0].payload if found else None

    def set_payload(self, spec: CollectionSpec, point_id: str, payload: dict[str, Any]) -> None:
        self._client.set_payload(spec.name, payload=payload, points=[point_id])

    def scroll(self, spec: CollectionSpec, flt: models.Filter | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        points, _ = self._client.scroll(spec.name, scroll_filter=flt, limit=limit, with_payload=True)
        return [{"id": str(p.id), **p.payload} for p in points]

    def _nearest_duplicate(self, spec: CollectionSpec, vector: list[float]) -> str | None:
        found = self._client.query_points(spec.name, query=vector, using=DENSE, limit=1, score_threshold=DEDUP_THRESHOLD).points
        return str(found[0].id) if found else None


@lru_cache(maxsize=1)
def get_semantic() -> SemanticStore:
    """Синглтон L3: QDRANT_URL или локальное хранилище на диске."""
    location = get_settings().resolved_qdrant_location
    client = QdrantClient(url=location) if location.startswith("http") else QdrantClient(path=location)
    atexit.register(client.close)
    return SemanticStore(client, get_embedder())
