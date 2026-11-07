"""Асинхронный клиент HTTP API деска."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import httpx

TIMEOUT_S = 60.0


class DeskBusy(RuntimeError):
    """У пользователя уже идёт анализ."""


class DeskClient:
    """Обёртка над эндпоинтами /v1/* для бота."""

    def __init__(self, base_url: str) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT_S)

    async def analyze(self, query: str, requester: str, as_of: date | None) -> str:
        payload = {"query": query, "requester": requester, "as_of_date": as_of.isoformat() if as_of else None}
        response = await self._http.post("/v1/analyze", json=payload)
        if response.status_code == 409:
            raise DeskBusy(response.json().get("detail", ""))
        response.raise_for_status()
        return response.json()["run_id"]

    async def runs(self, requester: str, limit: int = 10) -> list[dict[str, Any]]:
        return await self._get("/v1/runs", requester=requester, limit=limit)

    async def run(self, run_id: str) -> dict[str, Any]:
        return await self._get(f"/v1/runs/{run_id}")

    async def report(self, run_id: str) -> dict[str, Any]:
        return await self._get(f"/v1/runs/{run_id}/report")

    async def report_pdf(self, run_id: str) -> bytes:
        response = await self._http.get(f"/v1/runs/{run_id}/report.pdf")
        response.raise_for_status()
        return response.content

    async def predictions(self, *, requester: str | None = None, run_id: str | None = None, ticker: str | None = None,
                          verified_since: datetime | None = None) -> list[dict[str, Any]]:
        params = {"requester": requester, "run_id": run_id, "ticker": ticker,
                  "verified_since": verified_since.isoformat() if verified_since else None}
        return await self._get("/v1/predictions", **{k: v for k, v in params.items() if v is not None})

    async def metrics(self) -> dict[str, Any]:
        return await self._get("/v1/metrics/summary")

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, **params: Any) -> Any:
        response = await self._http.get(path, params=params)
        response.raise_for_status()
        return response.json()
