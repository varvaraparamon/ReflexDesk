"""Новости: Google News RSS, дедуп по url, ≤ 20 статей, срез ≤ as_of."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import httpx

from reflexdesk.cache import get_cache, make_key
from reflexdesk.tools.base import (
    Document,
    ToolError,
    ToolResult,
    as_of,
    clamp_date,
    tool,
)

MAX_ARTICLES = 20
NEWS_TTL_S = 3600
GOOGLE_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
_TAGS = re.compile(r"<[^>]+>")


def _google(query: str, start: date, end: date) -> list[dict]:
    q = quote_plus(f"{query} after:{start.isoformat()} before:{(end + timedelta(days=1)).isoformat()}")
    feed = feedparser.parse(httpx.get(GOOGLE_RSS.format(q=q), timeout=20, follow_redirects=True).text)
    items = []
    for entry in feed.entries:
        published = parsedate_to_datetime(entry.published) if entry.get("published") else None
        items.append({
            "title": entry.title,
            "url": entry.link,
            "published_at": published.isoformat() if published else None,
            "summary": _TAGS.sub(" ", entry.get("summary", "")).strip(),
        })
    return items


def dedupe(items: list[dict], cutoff: datetime, limit: int = MAX_ARTICLES) -> list[dict]:
    """Уникальные по url статьи не позже cutoff, от свежих к старым."""
    seen: set[str] = set()
    result = []
    for item in sorted(items, key=lambda i: i["published_at"] or "", reverse=True):
        published = datetime.fromisoformat(item["published_at"]) if item["published_at"] else None
        if item["url"] in seen or (published and published > cutoff):
            continue
        seen.add(item["url"])
        result.append(item)
    return result[:limit]


@tool("get_news", "новости по запросу за период: заголовки, ссылки, аннотации")
def get_news(query: str, start: str | None = None, end: str | None = None) -> ToolResult:
    """Новости за период [start, end] (по умолчанию — 30 дней до среза)."""
    end_d = clamp_date(end)
    start_d = clamp_date(start) if start else end_d - timedelta(days=30)
    cache = get_cache("news")
    key = make_key(query, start_d, end_d)
    items = cache.get(key)
    if items is None:
        items = _google(query, start_d, end_d)
        cache.set(key, items, NEWS_TTL_S)
    cutoff = as_of() if as_of().tzinfo else as_of().replace(tzinfo=UTC)
    articles = dedupe(items, cutoff)
    if not articles:
        raise ToolError(f"no news for {query!r} in {start_d}..{end_d}")
    documents = [
        Document(
            title=a["title"], text=a["summary"] or a["title"], url=a["url"],
            published_at=datetime.fromisoformat(a["published_at"]) if a["published_at"] else None,
        )
        for a in articles
    ]
    return ToolResult(tool="get_news", data={"count": len(articles)}, documents=documents)
