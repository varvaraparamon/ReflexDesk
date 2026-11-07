"""Рыночные данные: OHLCV из yfinance с фолбэком на Stooq, кэш 1 ч, срез ≤ as_of."""

from __future__ import annotations

import io
import warnings
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from reflexdesk.cache import get_cache, make_key
from reflexdesk.schemas import Evidence
from reflexdesk.tools.base import ToolError, ToolResult, clamp_date, tool

PRICE_TTL_S = 3600
YAHOO_URL = "https://finance.yahoo.com/quote/{ticker}/history"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i={interval}"
_STOOQ_INTERVALS = {"1d": "d", "1wk": "w", "1mo": "m"}


def load_prices(ticker: str, start: date, end: date, interval: str = "1d") -> tuple[pd.DataFrame, str]:
    """OHLCV + adj_close за [start, end]; возвращает (frame, source_url)."""
    cache = get_cache("prices")
    key = make_key(ticker, start, end, interval)
    cached = cache.get(key)
    if cached is not None:
        return _frame_from_records(cached["rows"]), cached["source"]
    try:
        frame, source = _from_yahoo(ticker, start, end, interval), YAHOO_URL.format(ticker=ticker)
    except Exception:
        frame, source = _from_stooq(ticker, start, end, interval), STOOQ_URL.format(
            symbol=_stooq_symbol(ticker), interval=_STOOQ_INTERVALS.get(interval, "d")
        )
    if frame.empty:
        raise ToolError(f"no price data for {ticker} in {start}..{end}")
    cache.set(key, {"rows": _records(frame), "source": source}, PRICE_TTL_S)
    return frame, source


def close_on(ticker: str, day: date, lookback_days: int = 10) -> tuple[float, date, str]:
    """Последняя цена закрытия (adjusted) не позже day."""
    frame, source = load_prices(ticker, day - timedelta(days=lookback_days), day)
    last = frame.iloc[-1]
    return float(last["adj_close"]), frame.index[-1].date(), source


def total_return(ticker: str, start: date, end: date) -> float:
    """Доходность по adjusted close между двумя датами."""
    p0, _, _ = close_on(ticker, start)
    p1, _, _ = close_on(ticker, end)
    return p1 / p0 - 1.0


def _from_yahoo(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.Ticker(ticker).history(
            start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval=interval, auto_adjust=False
        )
    if raw.empty:
        raise ToolError("yahoo returned empty frame")
    frame = raw.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return frame[["open", "high", "low", "close", "adj_close", "volume"]]


def _from_stooq(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    url = STOOQ_URL.format(symbol=_stooq_symbol(ticker), interval=_STOOQ_INTERVALS.get(interval, "d"))
    response = httpx.get(url, timeout=20)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), parse_dates=["Date"], index_col="Date")
    frame = frame.rename(columns=str.lower)
    frame["adj_close"] = frame["close"]
    frame = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    return frame[["open", "high", "low", "close", "adj_close", "volume"]]


def _stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower().replace('-', '.')}.us"


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{"date": idx.date().isoformat(), **row} for idx, row in frame.astype(float).to_dict("index").items()]


def _frame_from_records(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop("date"))
    return frame


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    """Ключевые статистики ряда: последняя цена, доходности, волатильность, диапазон 52 нед."""
    close = frame["adj_close"]
    daily = close.pct_change().dropna()
    last = float(close.iloc[-1])

    def ret(days: int) -> float | None:
        cutoff = close.index[-1] - pd.Timedelta(days=days)
        window = close.loc[close.index >= cutoff]
        return float(last / window.iloc[0] - 1) if len(window) > 1 else None

    year = close.loc[close.index >= close.index[-1] - pd.Timedelta(days=365)]
    stats = {
        "last_close": last,
        "return_1m": ret(30),
        "return_3m": ret(90),
        "return_1y": ret(365),
        "volatility_annual": float(daily.std() * np.sqrt(252)) if len(daily) > 5 else None,
        "high_52w": float(year.max()),
        "low_52w": float(year.min()),
        "avg_volume_3m": float(frame["volume"].iloc[-63:].mean()),
    }
    return {k: round(v, 6) for k, v in stats.items() if v is not None}


_LABELS = {
    "last_close": "Цена закрытия (adj)",
    "return_1m": "Доходность за 1 мес",
    "return_3m": "Доходность за 3 мес",
    "return_1y": "Доходность за 1 год",
    "volatility_annual": "Годовая волатильность",
    "high_52w": "Максимум 52 нед",
    "low_52w": "Минимум 52 нед",
    "avg_volume_3m": "Средний объём за 3 мес",
}


@tool("get_market_data", "котировки OHLCV, доходности, волатильность, диапазон 52 недель")
def get_market_data(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d") -> ToolResult:
    """OHLCV + adjusted close за период (не позже as_of)."""
    end_d = clamp_date(end)
    start_d = clamp_date(start) if start else end_d - timedelta(days=400)
    if start_d >= end_d:
        raise ToolError("start must be before end")
    frame, source = load_prices(ticker.upper(), start_d, end_d, interval)
    stats = summarize(frame)
    as_of_label = frame.index[-1].date().isoformat()
    evidence = [
        Evidence(fact=f"{_LABELS[k]} {ticker.upper()} на {as_of_label}", value=v, source_url=source, tool="get_market_data")
        for k, v in stats.items()
    ]
    weekly = frame["adj_close"].resample("W").last().dropna()
    return ToolResult(
        tool="get_market_data",
        data={"stats": stats, "weekly_close": {d.date().isoformat(): round(float(v), 4) for d, v in weekly.items()}},
        evidence=evidence,
    )
