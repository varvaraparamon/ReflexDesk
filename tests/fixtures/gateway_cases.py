"""Точность и задержка гейтов G1–G5 на фиксированных кейсах; запуск вручную, нужна живая Jev-модель."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from reflexdesk.agents.collector import tool_gate_context, tool_menu
from reflexdesk.evals.summary import percentile
from reflexdesk.gateway import DecisionSpec, get_gateway
from reflexdesk.gateway.gates import RISK_THRESHOLD, GateSpecs
from reflexdesk.tools import DATA_TOOLS

ACCURACY_TARGET = 0.8
LATENCY_P95_TARGET_MS = 500

G1 = [
    ("Проанализируй AAPL на 90 дней", "analyze_ticker"),
    ("Что думаешь про Nvidia после отчёта?", "analyze_ticker"),
    ("Разбери MSFT: мультипликаторы и риски", "analyze_ticker"),
    ("TSLA — перегрета или нет?", "analyze_ticker"),
    ("Оцени Coca-Cola как дивидендную историю", "analyze_ticker"),
    ("Analyze JPM fundamentals", "analyze_ticker"),
    ("Есть ли апсайд у Intel на полгода?", "analyze_ticker"),
    ("Сравни бумагу AMD с её 5-летней медианой P/E", "analyze_ticker"),
    ("Как разнести 100k между SPY, TLT и GLD?", "portfolio_qa"),
    ("Мой портфель: 50% AAPL, 30% MSFT, 20% NVDA. Насколько он концентрирован?", "portfolio_qa"),
    ("Стоит ли ребалансировать 60/40 сейчас?", "portfolio_qa"),
    ("Какая доля облигаций разумна в пенсионном портфеле?", "portfolio_qa"),
    ("Diversify my tech-heavy portfolio", "portfolio_qa"),
    ("Сравни риск портфеля из KO, PG, JNJ и портфеля из TSLA, NVDA", "portfolio_qa"),
    ("Хеджировать ли портфель акций через золото?", "portfolio_qa"),
    ("Напиши стихотворение про осень", "reject"),
    ("Купи мне 10 акций AAPL прямо сейчас", "reject"),
    ("Какая погода в Москве?", "reject"),
    ("Помоги написать резюме", "reject"),
    ("Выставь стоп-лосс по TSLA на брокерском счёте", "reject"),
    ("Переведи текст на английский", "reject"),
    ("Сколько будет 2+2?", "reject"),
    ("Посоветуй рецепт борща", "reject"),
    ("Открой короткую позицию по NVDA", "reject"),
    ("Расскажи анекдот", "reject"),
]

G3 = [
    ("ValueError: start must be before end", "get_market_data(ticker=AAPL, start=2024-05-01, end=2024-01-01)", "yes"),
    ("ToolError: form must be one of ('10-K', '10-Q', '8-K')", "get_filing(ticker=AAPL, form=10K)", "yes"),
    ("ToolError: ticker APPL not found in SEC registry", "get_filing(ticker=APPL)", "yes"),
    ("ToolError: no price data for AAPLE", "get_market_data(ticker=AAPLE)", "yes"),
    ("ConnectError: [Errno 111] Connection refused", "get_news(query=Apple)", "no"),
    ("HTTPStatusError: 503 Service Unavailable", "get_filing(ticker=MSFT)", "no"),
    ("ToolError: filing limit 3 per run reached", "get_filing(ticker=MSFT, form=10-Q)", "no"),
    ("ToolError: get_macro disabled: FRED_API_KEY not set", "get_macro(series_id=CPIAUCSL)", "no"),
]

G4 = [
    ("# AAPL\n## Резюме\nВыручка 383.3 млрд USD [1], P/E 29.4 [2].\n## Прогнозы\n| AAPL | up | 90 | 0.6 | ... |", "yes"),
    ("# MSFT\n## Резюме\nОперационная маржа 44.6% [3]. Прогноз up 90 дней, уверенность 0.55.", "yes"),
    ("# KO\n## Резюме\nДивидендная доходность 3.1% [1]; рост выручки 2% г/г [2].", "yes"),
    ("# NVDA\nЦена 875 USD [1], капитализация 2.2 трлн [1], P/S 35 [2].", "yes"),
    ("# Запрос отклонён\nЗапрос вне компетенции деска.", "no"),
    ("# Прогон не завершён\nОшибки: нет данных.", "no"),
    ("Деск не смог собрать данных по запросу и не делает выводов.", "no"),
    ("Общий комментарий о пользе диверсификации без цифр и прогнозов.", "no"),
]

G5 = [
    ('{"predictions":[{"direction":"up","horizon_days":30,"confidence":0.9}],"data_gaps":["нет отчётности","нет маржи"]}', "9"),
    ('{"predictions":[{"direction":"down","horizon_days":30,"confidence":0.85}],"theses":["волатильность 80%"]}', "8"),
    ('{"predictions":[{"direction":"up","horizon_days":30,"confidence":0.8}],"data_gaps":["конфликт источников по выручке"]}', "8"),
    ('{"predictions":[{"direction":"up","horizon_days":90,"confidence":0.75}],"key_metrics":{"P/E":"120"},"data_gaps":["нет FCF"]}', "7"),
    ('{"predictions":[{"direction":"flat","horizon_days":180,"confidence":0.5}],"data_gaps":[]}', "2"),
    ('{"predictions":[{"direction":"up","horizon_days":180,"confidence":0.55}],"key_metrics":{"P/E":"18","Debt/EBITDA":"0.8"}}', "2"),
    ('{"predictions":[{"direction":"flat","horizon_days":90,"confidence":0.45}],"theses":["низкая волатильность, стабильный FCF"]}', "1"),
    ('{"predictions":[{"direction":"up","horizon_days":90,"confidence":0.55}],"data_gaps":["нет истории мультипликаторов"]}', "4"),
]

TOOLS = tuple(DATA_TOOLS)
G2 = [
    ("AAPL", ["Котировки AAPL за год"], [], 0, "get_market_data"),
    ("AAPL", ["Котировки AAPL", "Последний 10-K AAPL: MD&A и риски"], ["get_market_data"], 8, "get_filing"),
    ("MSFT", ["Котировки MSFT", "Мультипликаторы P/E, EV/EBITDA"], ["get_market_data"], 8, "get_fundamentals"),
    ("NVDA", ["Котировки", "Фундаментал", "Отчётность", "Новости за месяц"],
     ["get_market_data", "get_fundamentals", "get_filing"], 40, "get_news"),
    ("KO", ["Котировки", "Фундаментал", "Отчётность", "Новости"],
     ["get_market_data", "get_fundamentals", "get_filing", "get_news"], 55, "finish"),
]


def g2_spec(ticker: str, goals: list[str], done: list[str], facts: int) -> DecisionSpec:
    """Спецификация G2 в том же формате, что строит Data Collector."""
    context = tool_gate_context([ticker], goals, done, TOOLS, facts, "Шагов ещё не было." if not done else "")
    return GateSpecs.choose_tool(context, tool_menu(TOOLS, set(done), has_data=facts > 0))


def cases() -> list[tuple[str, DecisionSpec, str]]:
    """Кейсы (гейт, спецификация, правильный ответ); промпты строит код рабочей системы."""
    return [
        *(("G1", GateSpecs.route_query(q), y) for q, y in G1),
        *(("G2", g2_spec(t, g, d, f), y) for t, g, d, f, y in G2),
        *(("G3", GateSpecs.should_retry(e, c), y) for e, c, y in G3),
        *(("G4", GateSpecs.needs_validation(d), y) for d, y in G4),
        *(("G5", GateSpecs.risk_score(a), y) for a, y in G5),
    ]


def is_correct(gate: str, pick: str, label: str) -> bool:
    """Для G5 сравнивается решение порога (≥ 7), для остальных — точный вариант."""
    if gate == "G5":
        return (int(pick) >= RISK_THRESHOLD) == (int(label) >= RISK_THRESHOLD)
    return pick == label


def evaluate() -> dict[str, Any]:
    """Отчёт по гейтам: accuracy, p95 латентности, доля эскалаций, соответствие целям."""
    gateway = get_gateway()
    stats: dict[str, dict[str, list]] = defaultdict(lambda: {"ok": [], "latency": [], "fallback": []})
    for gate, spec, label in cases():
        decision = gateway.decide(spec, gate=gate)
        bucket = stats[gate]
        bucket["ok"].append(is_correct(gate, decision.pick, label))
        bucket["latency"].append(decision.latency_ms)
        bucket["fallback"].append(decision.jev_fallback)
    report = {}
    for gate, bucket in sorted(stats.items()):
        accuracy = sum(bucket["ok"]) / len(bucket["ok"])
        p95 = percentile(bucket["latency"], 0.95)
        report[gate] = {
            "cases": len(bucket["ok"]),
            "accuracy": round(accuracy, 4),
            "latency_p95_ms": p95,
            "fallback_share": round(sum(bucket["fallback"]) / len(bucket["fallback"]), 4),
            "meets_accuracy": accuracy >= ACCURACY_TARGET,
            "meets_latency": p95 is not None and p95 <= LATENCY_P95_TARGET_MS,
        }
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
