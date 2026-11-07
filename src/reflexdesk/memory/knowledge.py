"""Доменные обёртки L3: документы (чанки), профили тикеров (версии), уроки (с superseded_by)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qdrant_client import models

from reflexdesk.memory.semantic import (
    DOCUMENTS,
    LESSONS,
    PROFILES,
    Hit,
    MemoryRecord,
    SemanticStore,
    profile_id,
)
from reflexdesk.schemas import Lesson, utcnow

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
MIN_LESSON_SCORE = 0.1


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Чанки фиксированного размера с перекрытием по границе абзаца, если возможно."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        cut = text.rfind("\n", start + size // 2, end)
        end = cut if cut > 0 and end < len(text) else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


class DocumentMemory:
    """Коллекция documents: чанки отчётности и новостей."""

    def __init__(self, store: SemanticStore) -> None:
        self._store = store

    def ingest(self, text: str, *, url: str, title: str, ticker: str | None, published_at: datetime | None, kind: str) -> list[str]:
        records = [
            MemoryRecord(text=chunk, ticker=ticker, published_at=published_at, meta={"url": url, "title": title, "kind": kind})
            for chunk in chunk_text(text)
        ]
        return self._store.upsert(DOCUMENTS, records)

    def search(self, query: str, *, as_of: datetime, ticker: str | None = None, top_k: int = 8) -> list[Hit]:
        return self._store.search(DOCUMENTS, query, as_of=as_of, ticker=ticker, top_k=top_k)


class ProfileMemory:
    """Коллекция ticker_profiles: агрегированный профиль с инкрементом версии."""

    def __init__(self, store: SemanticStore) -> None:
        self._store = store

    def get(self, ticker: str) -> dict[str, Any] | None:
        return self._store.get(PROFILES, profile_id(ticker))

    def update(self, ticker: str, *, sector: str | None, metrics: dict[str, Any], stance: str, as_of: datetime) -> int:
        """Merge новых метрик в профиль; возвращает новую версию."""
        current = self.get(ticker) or {}
        merged = {**current.get("metrics", {}), **metrics}
        version = int(current.get("version", 0)) + 1
        sector = sector or current.get("sector")
        text = f"{ticker.upper()} ({sector or 'сектор н/д'}). Позиция деска: {stance}. " + "; ".join(
            f"{k}: {v}" for k, v in merged.items()
        )
        self._store.upsert(
            PROFILES,
            [MemoryRecord(
                id=profile_id(ticker), text=text, ticker=ticker, sector=sector, published_at=as_of,
                meta={"metrics": merged, "stance": stance, "version": version, "updated_at": utcnow().isoformat()},
            )],
        )
        return version


class LessonMemory:
    """Коллекция lessons: уроки Reflector, retrieval по тикеру/сектору."""

    def __init__(self, store: SemanticStore) -> None:
        self._store = store

    def add(self, lessons: list[Lesson]) -> list[str]:
        records = [
            MemoryRecord(
                id=lesson.lesson_id, text=lesson.text, ticker=lesson.ticker, sector=lesson.sector,
                published_at=lesson.created_at,
                meta={"lesson": lesson.model_dump(mode="json"), "error_type": lesson.error_type, "superseded_by": None},
            )
            for lesson in lessons
        ]
        return self._store.upsert(LESSONS, records)

    def relevant(self, query: str, *, as_of: datetime, ticker: str | None, sector: str | None, top_k: int = 5) -> list[Hit]:
        """Актуальные (не замещённые) уроки, релевантные тикеру/сектору."""
        active = [models.IsEmptyCondition(is_empty=models.PayloadField(key="superseded_by"))]
        hits = self._store.search(LESSONS, query, as_of=as_of, ticker=ticker, sector=sector, top_k=top_k, extra_filter=active)
        return [h for h in hits if h.score >= MIN_LESSON_SCORE]

    def active(self, ticker: str | None = None) -> list[dict[str, Any]]:
        must: list[models.Condition] = [models.IsEmptyCondition(is_empty=models.PayloadField(key="superseded_by"))]
        if ticker:
            must.append(models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper())))
        return self._store.scroll(LESSONS, models.Filter(must=must))

    def supersede(self, lesson_id: str, by: str) -> None:
        self._store.set_payload(LESSONS, lesson_id, {"superseded_by": by})
