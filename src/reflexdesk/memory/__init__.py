"""Подсистема памяти: L1 чекпоинты, L2 журнал, L3 семантическая память."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from reflexdesk.memory.episodic import EpisodicStore, get_episodic
from reflexdesk.memory.knowledge import DocumentMemory, LessonMemory, ProfileMemory


@dataclass(frozen=True)
class Knowledge:
    """Доступ к трём коллекциям L3."""

    documents: DocumentMemory
    profiles: ProfileMemory
    lessons: LessonMemory


@lru_cache(maxsize=1)
def get_knowledge() -> Knowledge:
    """Синглтон L3-фасада (ленивая загрузка моделей эмбеддингов)."""
    from reflexdesk.memory.semantic import get_semantic

    store = get_semantic()
    return Knowledge(DocumentMemory(store), ProfileMemory(store), LessonMemory(store))


__all__ = ["EpisodicStore", "Knowledge", "get_episodic", "get_knowledge"]
