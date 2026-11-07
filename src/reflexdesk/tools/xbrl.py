"""Point-in-time выборка показателей из XBRL companyfacts (только поданное не позже среза)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from reflexdesk.tools.edgar import FILING_INDEX_URL

CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "depreciation": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "current_debt": ("LongTermDebtCurrent", "DebtCurrent"),
    "commercial_paper": ("CommercialPaper",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "shares_outstanding": ("EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"),
}
FLOW_ITEMS = {"revenue", "gross_profit", "operating_income", "net_income", "eps_diluted", "operating_cash_flow", "capex", "depreciation"}
ANNUAL_DAYS = (350, 380)


@dataclass(frozen=True)
class FactPoint:
    """Значение показателя с привязкой к филингу."""

    name: str
    concept: str
    value: float
    end: date
    filed: date
    form: str
    accession: str
    cik: str

    @property
    def url(self) -> str:
        return FILING_INDEX_URL.format(cik=int(self.cik), accession=self.accession.replace("-", ""))


def _units(facts: dict, concept: str) -> list[dict]:
    for taxonomy in ("us-gaap", "dei"):
        node = facts.get("facts", {}).get(taxonomy, {}).get(concept)
        if node:
            for unit in ("USD", "USD/shares", "shares"):
                if unit in node["units"]:
                    return node["units"][unit]
    return []


def _is_annual(entry: dict) -> bool:
    if "start" not in entry:
        return False
    days = (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days
    return ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]


def latest(facts: dict, cik: str, name: str, until: date, accession: str | None = None) -> FactPoint | None:
    """Последнее значение показателя, поданное не позже until (годовое для потоковых статей)."""
    best: FactPoint | None = None
    for concept in CONCEPTS[name]:
        for entry in _units(facts, concept):
            filed = date.fromisoformat(entry["filed"])
            if filed > until or (accession and entry.get("accn") != accession):
                continue
            if name in FLOW_ITEMS and not _is_annual(entry):
                continue
            point = FactPoint(
                name=name, concept=concept, value=float(entry["val"]), end=date.fromisoformat(entry["end"]),
                filed=filed, form=entry.get("form", ""), accession=entry.get("accn", ""), cik=cik,
            )
            if best is None or (point.end, point.filed) > (best.end, best.filed):
                best = point
        if best is not None:
            return best
    return None


def snapshot(facts: dict, cik: str, until: date, accession: str | None = None) -> dict[str, FactPoint]:
    """Все доступные показатели на дату среза."""
    points = {name: latest(facts, cik, name, until, accession) for name in CONCEPTS}
    return {k: v for k, v in points.items() if v is not None}
