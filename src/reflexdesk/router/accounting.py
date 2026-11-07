"""Расчёт стоимости и учёт расходов по прогонам."""

from __future__ import annotations

import threading
from collections import defaultdict

from reflexdesk.router.config import Price


def compute_cost(price: Price, tokens_in: int, cached_in: int, tokens_out: int) -> float:
    """Стоимость вызова в USD по токенам."""
    fresh_in = max(tokens_in - cached_in, 0)
    return (fresh_in * price.input + cached_in * price.cached_input + tokens_out * price.output) / 1_000_000


class SpendLedger:
    """Потокобезопасный учёт расходов по (run_id, tier)."""

    def __init__(self) -> None:
        self._spent: dict[tuple[str, str], float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, run_id: str | None, tier: str, cost: float) -> None:
        if run_id is None:
            return
        with self._lock:
            self._spent[(run_id, tier)] += cost

    def spent(self, run_id: str | None, tier: str) -> float:
        if run_id is None:
            return 0.0
        with self._lock:
            return self._spent.get((run_id, tier), 0.0)

    def total(self, run_id: str) -> float:
        with self._lock:
            return sum(v for (rid, _), v in self._spent.items() if rid == run_id)

    def release(self, run_id: str) -> None:
        with self._lock:
            for key in [k for k in self._spent if k[0] == run_id]:
                del self._spent[key]
