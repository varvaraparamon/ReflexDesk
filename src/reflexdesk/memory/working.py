"""L1 working memory: чекпоинтер LangGraph (Postgres или SQLite) и retention 30 дней."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from reflexdesk import schemas
from reflexdesk.config import get_settings
from reflexdesk.memory.episodic import EpisodicStore
from reflexdesk.schemas import utcnow

RETENTION = timedelta(days=30)


def serializer() -> JsonPlusSerializer:
    """Сериализатор состояния с явным whitelist артефактов деска."""
    allowed = [
        (schemas.__name__, name)
        for name, obj in vars(schemas).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def open_checkpointer() -> BaseCheckpointSaver:
    """Чекпоинтер по CHECKPOINT_URL: postgresql://... -> PostgresSaver, иначе файл SQLite."""
    target = get_settings().resolved_checkpoint_url
    if target.startswith("postgres"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(target, kwargs={"autocommit": True, "prepare_threshold": 0}, max_size=10, open=True)
        saver = PostgresSaver(pool, serde=serializer())
        saver.setup()
        return saver
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(sqlite3.connect(target, check_same_thread=False), serde=serializer())
    saver.setup()
    return saver


def purge_expired(checkpointer: BaseCheckpointSaver, episodic: EpisodicStore) -> int:
    """Удаляет чекпоинты прогонов старше 30 дней; L2 не трогает."""
    expired = episodic.runs_older_than(utcnow() - RETENTION)
    for run_id in expired:
        checkpointer.delete_thread(run_id)
    return len(expired)
