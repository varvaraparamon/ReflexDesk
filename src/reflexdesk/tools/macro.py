"""Макроряды FRED; доступен только при наличии FRED_API_KEY."""

from __future__ import annotations

from datetime import timedelta

import httpx

from reflexdesk.config import get_settings
from reflexdesk.schemas import Evidence
from reflexdesk.tools.base import ToolError, ToolResult, clamp_date, tool

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_PAGE = "https://fred.stlouisfed.org/series/{series_id}"


@tool("get_macro", "макроэкономический ряд FRED (ставки, инфляция, безработица)")
def get_macro(series_id: str, start: str | None = None, end: str | None = None) -> ToolResult:
    """Наблюдения ряда FRED за период, не позже as_of."""
    settings = get_settings()
    if not settings.fred_enabled:
        raise ToolError("get_macro disabled: FRED_API_KEY not set")
    end_d = clamp_date(end)
    start_d = clamp_date(start) if start else end_d - timedelta(days=365 * 2)
    response = httpx.get(
        FRED_URL,
        params={
            "series_id": series_id, "api_key": settings.fred_api_key, "file_type": "json",
            "observation_start": start_d.isoformat(), "observation_end": end_d.isoformat(),
            "realtime_start": end_d.isoformat(), "realtime_end": end_d.isoformat(),
        },
        timeout=20,
    )
    response.raise_for_status()
    observations = [o for o in response.json().get("observations", []) if o["value"] != "."]
    if not observations:
        raise ToolError(f"no observations for {series_id}")
    url = FRED_PAGE.format(series_id=series_id)
    last = observations[-1]
    evidence = [Evidence(fact=f"{series_id} на {last['date']}", value=float(last["value"]), source_url=url, tool="get_macro")]
    series = {o["date"]: float(o["value"]) for o in observations}
    return ToolResult(tool="get_macro", data={"series_id": series_id, "series": series}, evidence=evidence)
