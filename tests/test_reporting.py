"""Разбор отчёта и faithfulness-скрипт."""

from __future__ import annotations

from types import SimpleNamespace

from reflexdesk.evals.golden import GoldenSuite
from reflexdesk.reporting.claims import computed_numbers, extract_claims, faithfulness
from reflexdesk.reporting.parse import parse_predictions
from reflexdesk.reporting.render import DISCLAIMER, ensure_disclaimer
from reflexdesk.schemas import Evidence

REPORT = """# AAPL: тезис
_Дата среза данных: 2024-03-01_

## Резюме
Цена 179.66 [1], выручка 383.29 млрд USD [2], рост 2.8% [2] за 3 мес. Отчёт 10-K от 28 октября.
Прибыль 99.8 млрд USD [3].

## Ключевые метрики
| Метрика | Значение | Источник |
|---|---|---|
| P/E | 27.9 | [2] |
| FCF-маржа | 25.8% (расчёт: 99.58 / 383.29) | [2] |

## Прогнозы
| Тикер | Направление | Горизонт, дней | Уверенность | Тезис |
|---|---|---|---|---|
| AAPL | up | 90 | 0.6 | Импульс |
| AAPL | рост | 30 | 55% | Тезис 2 |

## Источники
1. https://a
2. https://b
"""

EVIDENCE = [
    Evidence(fact="Цена закрытия", value=179.66, source_url="https://a"),
    Evidence(fact="Выручка, USD", value=383285000000, source_url="https://b"),
    Evidence(fact="Рост", value=0.028, source_url="https://b"),
    Evidence(fact="P/E", value="27.93", source_url="https://b"),
]


def test_parse_predictions_table():
    predictions, errors = parse_predictions(REPORT)
    assert not errors
    assert [(p.direction, p.horizon_days, p.confidence) for p in predictions] == [("up", 90, 0.6), ("up", 30, 0.55)]


def test_parse_predictions_missing_section():
    predictions, errors = parse_predictions("# x\n## Резюме\nтекст")
    assert not predictions and errors


def test_claims_skip_dates_periods_refs_and_tables():
    raws = [c.raw for c in extract_claims(REPORT)]
    assert "179.66" in raws and "99.8 млрд" in raws
    assert not any(r.startswith("28") or r == "3" or r == "10" for r in raws)
    assert all(r not in ("90", "0.6") for r in raws)


def test_faithfulness_scales_and_percents():
    report = faithfulness(REPORT, EVIDENCE)
    unsupported = {c.raw for c in report.unsupported}
    assert "99.8 млрд" in unsupported
    assert "383.29 млрд" not in unsupported and "2.8%" not in unsupported and "27.9" not in unsupported
    assert report.derived >= 1
    assert 0 < report.ratio < report.grounded <= 1


def test_computed_numbers_count_as_support():
    strict = faithfulness(REPORT, EVIDENCE)
    with_computations = faithfulness(REPORT, EVIDENCE, computed_numbers(["Выручка сервисов: 99.8 млрд"]))
    assert with_computations.supported > strict.supported
    assert "99.8 млрд" not in {c.raw for c in with_computations.unsupported}


def test_disclaimer_added_once():
    text = ensure_disclaimer("# r")
    assert DISCLAIMER in text and ensure_disclaimer(text) == text


def test_golden_skips_quality_scores_of_failed_run():
    suite = GoldenSuite.__new__(GoldenSuite)
    suite.service = SimpleNamespace(state=lambda _: {}, services=SimpleNamespace(episodic=None))
    scores = suite.evaluate({}, {"run_id": "r", "cost_usd": 0.01, "status": "failed"})
    assert scores == {"cost_usd": 0.01, "completed": 0.0}
