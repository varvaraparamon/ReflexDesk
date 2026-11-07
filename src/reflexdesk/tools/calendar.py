"""Биржевой календарь NYSE."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from reflexdesk.tools.base import ToolResult, clamp_date, tool

CALENDAR_URL = "https://www.nyse.com/markets/hours-calendars"


@lru_cache(maxsize=1)
def _nyse():
    import pandas_market_calendars as mcal

    return mcal.get_calendar("NYSE")


def trading_days(start: date, end: date) -> list[date]:
    """Торговые дни NYSE в диапазоне."""
    schedule = _nyse().schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [ts.date() for ts in schedule.index]


def next_trading_day(day: date) -> date:
    """Ближайший торговый день, не раньше day."""
    days = trading_days(day, day + timedelta(days=10))
    return days[0]


@tool("get_market_calendar", "торговый ли день на NYSE и ближайшее открытие")
def get_market_calendar(day: str | None = None) -> ToolResult:
    """is_open и next_open для даты (по умолчанию — дата среза)."""
    target = clamp_date(day)
    upcoming = next_trading_day(target)
    is_open = upcoming == target
    next_open = upcoming if not is_open else next_trading_day(target + timedelta(days=1))
    return ToolResult(
        tool="get_market_calendar",
        data={"date": target.isoformat(), "is_open": is_open, "next_open": next_open.isoformat(), "source": CALENDAR_URL},
    )
