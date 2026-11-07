"""Фундаментальные показатели: live — yfinance info; исторический срез — SEC XBRL point-in-time."""

from __future__ import annotations

from reflexdesk.schemas import Evidence
from reflexdesk.tools import xbrl
from reflexdesk.tools.base import ToolResult, as_of_date, is_historical, tool
from reflexdesk.tools.edgar import get_edgar
from reflexdesk.tools.market import close_on

YAHOO_STATS_URL = "https://finance.yahoo.com/quote/{ticker}/key-statistics"

YAHOO_FIELDS = {
    "trailingPE": "P/E (TTM)",
    "forwardPE": "P/E (forward)",
    "enterpriseToEbitda": "EV/EBITDA",
    "priceToBook": "P/B",
    "marketCap": "Рыночная капитализация, USD",
    "enterpriseValue": "Enterprise value, USD",
    "freeCashflow": "Free cash flow (TTM), USD",
    "totalDebt": "Совокупный долг, USD",
    "totalCash": "Денежные средства, USD",
    "ebitda": "EBITDA (TTM), USD",
    "totalRevenue": "Выручка (TTM), USD",
    "revenueGrowth": "Рост выручки г/г",
    "grossMargins": "Валовая маржа",
    "operatingMargins": "Операционная маржа",
    "profitMargins": "Чистая маржа",
    "dividendYield": "Дивидендная доходность",
    "beta": "Бета",
}

XBRL_LABELS = {
    "revenue": "Выручка (годовая), USD",
    "gross_profit": "Валовая прибыль (годовая), USD",
    "operating_income": "Операционная прибыль (годовая), USD",
    "net_income": "Чистая прибыль (годовая), USD",
    "eps_diluted": "EPS diluted (годовой), USD",
    "operating_cash_flow": "Операционный денежный поток (годовой), USD",
    "capex": "Капзатраты (годовые), USD",
    "depreciation": "Амортизация (годовая), USD",
    "long_term_debt": "Долгосрочный долг, USD",
    "current_debt": "Краткосрочный долг, USD",
    "commercial_paper": "Коммерческие бумаги, USD",
    "cash": "Денежные средства, USD",
    "shares_outstanding": "Акций в обращении",
}


def derived_ratios(points: dict[str, xbrl.FactPoint], price: float) -> dict[str, float]:
    """Мультипликаторы из point-in-time фактов и цены на дату среза."""
    v = {k: p.value for k, p in points.items()}
    out: dict[str, float] = {}
    debt = sum(v.get(k, 0.0) for k in ("long_term_debt", "current_debt", "commercial_paper"))
    ebitda = v["operating_income"] + v["depreciation"] if {"operating_income", "depreciation"} <= v.keys() else None
    fcf = v["operating_cash_flow"] - v.get("capex", 0.0) if "operating_cash_flow" in v else None
    mcap = price * v["shares_outstanding"] if "shares_outstanding" in v else None
    if v.get("eps_diluted", 0) > 0:
        out["P/E (годовой EPS)"] = price / v["eps_diluted"]
    if mcap:
        out["Рыночная капитализация, USD"] = mcap
        if ebitda and ebitda > 0:
            out["EV/EBITDA"] = (mcap + debt - v.get("cash", 0.0)) / ebitda
        if fcf is not None:
            out["FCF yield"] = fcf / mcap
    if fcf is not None:
        out["Free cash flow (годовой), USD"] = fcf
    if ebitda and ebitda > 0:
        out["EBITDA (годовая), USD"] = ebitda
        out["Debt/EBITDA"] = debt / ebitda
    if v.get("revenue"):
        for key, label in (("gross_profit", "Валовая маржа"), ("operating_income", "Операционная маржа"), ("net_income", "Чистая маржа")):
            if key in v:
                out[label] = v[key] / v["revenue"]
    return {k: round(x, 6) for k, x in out.items()}


def _live(ticker: str) -> ToolResult:
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    url = YAHOO_STATS_URL.format(ticker=ticker)
    evidence = [
        Evidence(fact=f"{label} {ticker}", value=info[key], source_url=url, tool="get_fundamentals")
        for key, label in YAHOO_FIELDS.items()
        if isinstance(info.get(key), (int, float))
    ]
    profile = {k: info.get(k) for k in ("sector", "industry", "longName", "country")}
    return ToolResult(tool="get_fundamentals", data={"profile": profile, "source": "yahoo"}, evidence=evidence)


def _point_in_time(ticker: str) -> ToolResult:
    cutoff = as_of_date()
    cik, facts = get_edgar().company_facts(ticker)
    points = xbrl.snapshot(facts, cik, cutoff)
    price, price_day, price_url = close_on(ticker, cutoff)
    evidence = [
        Evidence(
            fact=f"{XBRL_LABELS[name]} {ticker} (период до {p.end}, {p.form}, подано {p.filed})",
            value=p.value, source_url=p.url, tool="get_fundamentals",
        )
        for name, p in points.items()
    ]
    base_url = points["revenue"].url if "revenue" in points else price_url
    evidence += [
        Evidence(fact=f"{label} {ticker} на {price_day} (расчёт: XBRL + цена)", value=value, source_url=base_url, tool="get_fundamentals")
        for label, value in derived_ratios(points, price).items()
    ]
    return ToolResult(
        tool="get_fundamentals",
        data={"profile": {"entity": facts.get("entityName")}, "source": "sec_xbrl", "price": price},
        evidence=evidence,
    )


@tool("get_fundamentals", "мультипликаторы и финансовые показатели: P/E, EV/EBITDA, FCF, долг, маржи")
def get_fundamentals(ticker: str) -> ToolResult:
    """Фундаментал на дату среза; для исторических прогонов — только поданное в SEC до as_of."""
    ticker = ticker.upper()
    return _point_in_time(ticker) if is_historical() else _live(ticker)
