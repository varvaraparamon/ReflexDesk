"""Тул run_backtest: цены ≤ as_of подгружаются на стороне app, расчёт — в sandbox через python_exec."""

from __future__ import annotations

import inspect
import json
from datetime import date, timedelta

from reflexdesk.evals import backtest_core
from reflexdesk.tools.base import ToolError, ToolResult, clamp_date, tool
from reflexdesk.tools.market import load_prices
from reflexdesk.tools.python_exec import run_code


def _closes(ticker: str, start: date, end: date) -> dict[str, float]:
    frame, _ = load_prices(ticker, start, end)
    return {d.date().isoformat(): round(float(v), 6) for d, v in frame["adj_close"].items()}


def build_program(signals: list[dict], prices: dict, benchmark: dict) -> str:
    """Исходник ядра + данные литералами + вызов; переменная result — итог."""
    payload = json.dumps({"signals": signals, "prices": prices, "benchmark": benchmark})
    return f"{inspect.getsource(backtest_core)}\n\n_data = {payload}\nresult = backtest(_data['signals'], _data['prices'], _data['benchmark'])\n"


@tool("run_backtest", "бэктест сигналов (прогнозов) против бенчмарка: Sharpe, max drawdown, hit rate")
def run_backtest(signals: list[dict], benchmark: str = "SPY") -> ToolResult:
    """signals: [{ticker, direction, start, end}] — только исторические данные ≤ as_of."""
    if not signals:
        raise ToolError("no signals")
    cutoff = clamp_date(None)
    clean = [{**s, "end": min(date.fromisoformat(str(s["end"])[:10]), cutoff).isoformat(), "start": str(s["start"])[:10]} for s in signals]
    start = min(date.fromisoformat(s["start"]) for s in clean) - timedelta(days=5)
    end = max(date.fromisoformat(s["end"]) for s in clean)
    prices = {t: _closes(t, start, end) for t in {s["ticker"] for s in clean}}
    result = run_code(build_program(clean, prices, _closes(benchmark, start, end)))
    if result.error:
        raise ToolError(result.error)
    return ToolResult(tool="run_backtest", data=result.result)
