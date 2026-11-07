"""Key-value кэш на SQLite с TTL на запись."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    """Стабильный sha256-ключ по JSON-сериализуемым частям."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TTLCache:
    """Персистентный JSON-кэш; записи истекают по TTL."""

    def __init__(self, path: Path, namespace: str) -> None:
        self._path = path
        self._namespace = namespace
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, value TEXT, expires_at REAL,"
                " PRIMARY KEY (ns, key))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    def get(self, key: str) -> Any | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM kv WHERE ns = ? AND key = ?", (self._namespace, key)
            ).fetchone()
        if row is None or row[1] < time.time():
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any, ttl_s: float) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv (ns, key, value, expires_at) VALUES (?, ?, ?, ?)",
                (self._namespace, key, payload, time.time() + ttl_s),
            )


def get_cache(namespace: str) -> TTLCache:
    """Кэш в общей директории данных."""
    from reflexdesk.config import get_settings

    return TTLCache(get_settings().cache_dir / "cache.sqlite", namespace)
