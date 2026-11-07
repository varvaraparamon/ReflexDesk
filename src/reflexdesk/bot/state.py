"""Персистентные отметки уведомлений бота (что уже отправлено каждому чату)."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from reflexdesk.schemas import utcnow

RUNS_KEY = "runs_until"
VERIFIED_KEY = "verified_until"


class BotState:
    """Водяные знаки по чатам: до какого updated_at/verified_at уведомления уже отправлены."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def mark(self, chat_id: int, key: str) -> datetime:
        """Отметка чата; при первом обращении — текущий момент (история до запуска бота не рассылается)."""
        with self._lock:
            entry = self._data.setdefault(str(chat_id), {})
            if key not in entry:
                entry[key] = utcnow().isoformat()
                self._save()
            return datetime.fromisoformat(entry[key])

    def advance(self, chat_id: int, key: str, value: datetime) -> None:
        with self._lock:
            entry = self._data.setdefault(str(chat_id), {})
            if key not in entry or datetime.fromisoformat(entry[key]) < value:
                entry[key] = value.isoformat()
                self._save()

    def notified_status(self, chat_id: int, run_id: str) -> str | None:
        """Последний статус прогона, о котором уже сообщено."""
        with self._lock:
            return self._data.get(str(chat_id), {}).get(f"run:{run_id}")

    def remember_status(self, chat_id: int, run_id: str, status: str) -> None:
        with self._lock:
            self._data.setdefault(str(chat_id), {})[f"run:{run_id}"] = status
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        tmp.replace(self._path)
