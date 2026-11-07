"""Отчётность SEC: ключевые секции 10-K/10-Q/8-K и XBRL-факты филинга; ≤ 3 документа на прогон."""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from datetime import UTC, datetime, time

from reflexdesk.obs.context import current_run_id
from reflexdesk.schemas import Evidence
from reflexdesk.tools import xbrl
from reflexdesk.tools.base import Document, ToolError, ToolResult, as_of_date, tool
from reflexdesk.tools.edgar import FilingRef, get_edgar
from reflexdesk.tools.fundamentals import XBRL_LABELS

MAX_DOCS_PER_RUN = 3
SECTION_CHARS = 8000
FORMS = ("10-K", "10-Q", "8-K")

SECTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "10-K": {
        "Risk Factors": (r"item\s*1a\.?\s*risk\s*factors", r"item\s*1b\.?|item\s*2\.?\s*properties"),
        "MD&A": (r"item\s*7\.?\s*management.s\s*discussion", r"item\s*7a\.?|item\s*8\.?"),
    },
    "10-Q": {
        "MD&A": (r"item\s*2\.?\s*management.s\s*discussion", r"item\s*3\.?|item\s*4\.?"),
        "Risk Factors": (r"item\s*1a\.?\s*risk\s*factors", r"item\s*2\.?|item\s*5\.?|item\s*6\.?"),
    },
}

_usage: dict[str, int] = defaultdict(int)
_usage_lock = threading.Lock()


def _reserve_slot() -> None:
    run_id = current_run_id.get() or "adhoc"
    with _usage_lock:
        if _usage[run_id] >= MAX_DOCS_PER_RUN:
            raise ToolError(f"filing limit {MAX_DOCS_PER_RUN} per run reached")
        _usage[run_id] += 1


def release_run(run_id: str) -> None:
    """Сбрасывает счётчик документов прогона."""
    with _usage_lock:
        _usage.pop(run_id, None)


def extract_section(text: str, start_pattern: str, end_pattern: str, limit: int = SECTION_CHARS) -> str | None:
    """Самое длинное вхождение секции между заголовками (пропускает оглавление)."""
    best = ""
    for match in re.finditer(start_pattern, text, re.IGNORECASE):
        tail = text[match.start():]
        end = re.search(end_pattern, tail[200:], re.IGNORECASE)
        body = tail[: end.start() + 200] if end else tail[:limit * 3]
        if len(body) > len(best):
            best = body
    return best[:limit] if len(best) > 500 else None


def _choose(refs: list[FilingRef], year: int | None) -> FilingRef:
    if not refs:
        raise ToolError("no filings of this form before as_of")
    if year is not None:
        for ref in refs:
            if ref.filing_date.year == year or (ref.report_date and ref.report_date.year == year):
                return ref
    return refs[0]


@tool("get_filing", "отчётность SEC EDGAR (10-K/10-Q/8-K): MD&A, риск-факторы, XBRL-показатели")
def get_filing(ticker: str, form: str = "10-K", year: int | None = None) -> ToolResult:
    """Секции филинга и XBRL-факты этого филинга, поданного не позже as_of."""
    form = form.upper()
    if form not in FORMS:
        raise ToolError(f"form must be one of {FORMS}")
    ticker = ticker.upper()
    edgar = get_edgar()
    ref = _choose(edgar.filings(ticker, form, as_of_date()), year)
    _reserve_slot()
    text = edgar.document_text(ref)
    title = f"{ticker} {form} от {ref.filing_date}"
    published = datetime.combine(ref.filing_date, time.min, tzinfo=UTC)
    documents = []
    for section, (start, end) in SECTIONS.get(form, {}).items():
        body = extract_section(text, start, end)
        if body:
            documents.append(Document(title=f"{title}: {section}", text=body, url=ref.url, published_at=published))
    if not documents:
        documents.append(Document(title=title, text=text[:SECTION_CHARS], url=ref.url, published_at=published))

    evidence: list[Evidence] = []
    if form in ("10-K", "10-Q"):
        cik, facts = edgar.company_facts(ticker)
        for name, point in xbrl.snapshot(facts, cik, as_of_date(), accession=ref.accession).items():
            evidence.append(
                Evidence(fact=f"{XBRL_LABELS[name]} {ticker} ({form}, период до {point.end})", value=point.value,
                         source_url=ref.url, tool="get_filing")
            )
    return ToolResult(
        tool="get_filing",
        data={"form": form, "filing_date": ref.filing_date.isoformat(), "url": ref.url},
        evidence=evidence,
        documents=documents,
    )
