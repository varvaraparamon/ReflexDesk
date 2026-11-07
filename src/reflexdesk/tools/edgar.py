"""Клиент SEC EDGAR: CIK, submissions, документы, XBRL companyfacts; троттлинг ≤ 5 rps и кэш."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from html import unescape

import httpx

from reflexdesk.cache import get_cache
from reflexdesk.config import get_settings
from reflexdesk.tools.base import ToolError

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

DAY_S = 86400
MIN_INTERVAL_S = 0.2


@dataclass(frozen=True)
class FilingRef:
    """Ссылка на конкретный филинг."""

    cik: str
    form: str
    accession: str
    filing_date: date
    report_date: date | None
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=int(self.cik), accession=self.accession.replace("-", ""), document=self.primary_document)

    @property
    def index_url(self) -> str:
        return FILING_INDEX_URL.format(cik=int(self.cik), accession=self.accession.replace("-", ""))


class EdgarClient:
    """HTTP-клиент SEC с обязательным User-Agent и ограничением частоты."""

    def __init__(self, user_agent: str) -> None:
        self._client = httpx.Client(headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        self._lock = threading.Lock()
        self._last = 0.0
        self._cache = get_cache("edgar")

    def _get(self, url: str, ttl_s: float, as_json: bool = True):
        cached = self._cache.get(url)
        if cached is not None:
            return cached
        with self._lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        response = self._client.get(url)
        if response.status_code == 404:
            raise ToolError(f"SEC 404: {url}")
        response.raise_for_status()
        payload = response.json() if as_json else response.text
        self._cache.set(url, payload, ttl_s)
        return payload

    def cik(self, ticker: str) -> str:
        """10-значный CIK по тикеру."""
        table = self._get(TICKERS_URL, ttl_s=7 * DAY_S)
        wanted = ticker.upper().replace(".", "-")
        for row in table.values():
            if row["ticker"].upper() == wanted:
                return f"{int(row['cik_str']):010d}"
        raise ToolError(f"ticker {ticker} not found in SEC registry")

    def filings(self, ticker: str, form: str, until: date) -> list[FilingRef]:
        """Филинги формы, поданные не позже until, от новых к старым."""
        cik = self.cik(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik), ttl_s=DAY_S)["filings"]["recent"]
        refs = []
        for i, kind in enumerate(recent["form"]):
            if kind != form:
                continue
            filed = date.fromisoformat(recent["filingDate"][i])
            if filed > until:
                continue
            report = recent["reportDate"][i]
            refs.append(
                FilingRef(
                    cik=cik, form=kind, accession=recent["accessionNumber"][i], filing_date=filed,
                    report_date=date.fromisoformat(report) if report else None,
                    primary_document=recent["primaryDocument"][i],
                )
            )
        return sorted(refs, key=lambda r: r.filing_date, reverse=True)

    def document_text(self, ref: FilingRef) -> str:
        """Текст основного документа филинга без HTML."""
        return html_to_text(self._get(ref.url, ttl_s=30 * DAY_S, as_json=False))

    def company_facts(self, ticker: str) -> tuple[str, dict]:
        """XBRL companyfacts: (cik, facts)."""
        cik = self.cik(ticker)
        return cik, self._get(FACTS_URL.format(cik=cik), ttl_s=DAY_S)


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
_SPACES = re.compile(r"[ \t\xa0]+")
_BLANKS = re.compile(r"\n\s*\n+")


def html_to_text(html: str) -> str:
    """Грубое, но быстрое извлечение текста из HTML филинга."""
    text = _TAGS.sub("\n", html)
    text = unescape(text)
    text = _SPACES.sub(" ", text)
    return _BLANKS.sub("\n\n", text).strip()


@lru_cache(maxsize=1)
def get_edgar() -> EdgarClient:
    """Синглтон EDGAR-клиента."""
    return EdgarClient(get_settings().sec_user_agent)
