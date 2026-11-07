"""Извлечение числовых утверждений из отчёта и сверка с Evidence (faithfulness)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from reflexdesk.reporting.parse import PREDICTIONS_HEADING, SOURCES_HEADING, sections
from reflexdesk.schemas import Evidence

_NUMBER = re.compile(
    r"(?<![\w\[/.-])([-+−]?\d+(?:[.,  ]\d+)*)\s*"
    r"(%|п\.п\.|трлн|млрд|млн|тыс|trillion|billion|million|bn|[TBMK](?![a-zA-Z]))?",
    re.IGNORECASE,
)
_PERIOD_AFTER = re.compile(
    r"^\s*-?\s*(дн|мес|нед|год|лет|квартал|day|month|week|year|[KQ]\b|январ|феврал|март|апрел|ма[яй]|июн|июл|"
    r"август|сентябр|октябр|ноябр|декабр|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)
_DERIVED = re.compile(r"расч[её]т", re.IGNORECASE)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SOURCE_REF = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
_SCALE = {"трлн": 1e12, "trillion": 1e12, "t": 1e12, "млрд": 1e9, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "млн": 1e6, "million": 1e6, "m": 1e6, "тыс": 1e3, "k": 1e3}
EXCLUDED_SECTIONS = (PREDICTIONS_HEADING, SOURCES_HEADING)


@dataclass(frozen=True)
class NumberClaim:
    """Число из текста отчёта с контекстом."""

    raw: str
    value: float
    percent: bool
    context: str


@dataclass
class FaithfulnessReport:
    """Итог сверки чисел отчёта с Evidence."""

    total: int = 0
    supported: int = 0
    unsupported: list[NumberClaim] = field(default_factory=list)

    @property
    def derived(self) -> int:
        """Неподтверждённые числа, явно помеченные как расчёт."""
        return sum(1 for c in self.unsupported if _DERIVED.search(c.context))

    @property
    def ratio(self) -> float:
        return 1.0 if self.total == 0 else self.supported / self.total

    @property
    def grounded(self) -> float:
        """Доля чисел, подтверждённых Evidence или явно выведенных расчётом."""
        return 1.0 if self.total == 0 else (self.supported + self.derived) / self.total


def _to_float(raw: str) -> float | None:
    text = raw.replace("−", "-").replace(" ", " ")
    if re.fullmatch(r"[-+]?\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?", text):
        text = text.replace(" ", "").replace(",", "")
    else:
        text = text.replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _is_year(value: float, raw: str) -> bool:
    return raw.isdigit() and 1990 <= value <= 2100


def extract_claims(markdown: str) -> list[NumberClaim]:
    """Числа из разделов отчёта, кроме дат, годов, периодов, ссылок [n], прогнозов и источников."""
    claims: list[NumberClaim] = []
    for title, body in sections(markdown).items():
        if not title.startswith(EXCLUDED_SECTIONS):
            claims += text_claims(body)
    return claims


def text_claims(text: str) -> list[NumberClaim]:
    """Числа из произвольного текста, кроме дат, годов, периодов и ссылок [n]."""
    claims: list[NumberClaim] = []
    for line in _SOURCE_REF.sub(" ", _DATE.sub(" ", text)).splitlines():
        line_body = re.sub(r"^\s*(\d+\.|[-*])\s+", "", line)
        for match in _NUMBER.finditer(line_body):
            raw, unit = match.group(1), (match.group(2) or "").lower()
            value = _to_float(raw)
            if value is None or _is_year(value, raw) or _PERIOD_AFTER.match(line_body[match.end(1):]):
                continue
            percent = unit in ("%", "п.п.")
            scaled = value * _SCALE.get(unit, 1.0)
            start = max(match.start() - 60, 0)
            claims.append(NumberClaim(match.group(0).strip(), scaled, percent, line_body[start:match.end() + 40].strip()))
    return claims


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        match = _NUMBER.search(value)
        if match and (number := _to_float(match.group(1))) is not None:
            return number * _SCALE.get((match.group(2) or "").lower(), 1.0)
    return None


def _decimals(raw: str) -> int:
    match = re.search(r"[.,](\d+)\s*\D*$", raw)
    return len(match.group(1)) if match else 0


def _matches(claim: NumberClaim, reference: float) -> bool:
    candidates = [reference]
    if claim.percent:
        candidates.append(reference * 100)
    for value in candidates:
        tolerance = max(abs(value) * 0.01, 0.5 * 10 ** -_decimals(claim.raw))
        if abs(abs(claim.value) - abs(value)) <= tolerance:
            return True
        for scale in (1e3, 1e6, 1e9, 1e12):
            if abs(abs(claim.value) - abs(value) / scale) <= max(abs(value / scale) * 0.01, 0.05):
                return True
    return False


def evidence_numbers(evidence: list[Evidence]) -> list[float]:
    """Числовые значения из Evidence (поле value и числа в тексте факта)."""
    numbers = []
    for item in evidence:
        for source in (item.value, item.fact):
            number = _numeric(source)
            if number is not None:
                numbers.append(number)
    return numbers


def computed_numbers(computations: list[str]) -> list[float]:
    """Числа из результатов расчётов аналитика (python_exec)."""
    return [claim.value for claim in text_claims("\n".join(computations))]


def faithfulness(markdown: str, evidence: list[Evidence], extra: list[float] | None = None) -> FaithfulnessReport:
    """Доля чисел отчёта, подтверждённых Evidence (с допуском округления и масштабов)."""
    references = evidence_numbers(evidence) + list(extra or [])
    report = FaithfulnessReport()
    for claim in extract_claims(markdown):
        report.total += 1
        if any(_matches(claim, ref) for ref in references):
            report.supported += 1
        else:
            report.unsupported.append(claim)
    return report
