"""Фейковые роутер, гейты и тулы для интеграционных тестов графа."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from reflexdesk.gateway.spec import Decision
from reflexdesk.router.accounting import SpendLedger
from reflexdesk.router.results import LLMResult, StructuredResult
from reflexdesk.schemas import (
    AnalysisDraft,
    AnalystStep,
    DebateCase,
    DraftPrediction,
    EvidenceBatch,
    Plan,
    RiskReport,
    SubTask,
    ValidationResult,
)

REPORT = """# TEST: тезис
## Резюме
Цена 100.5 [1].
## Тезис Bull
## Тезис Bear
## Ключевые метрики
## Риски
## Прогнозы
| Тикер | Направление | Горизонт, дней | Уверенность | Тезис |
|---|---|---|---|---|
| TEST | up | 30 | 0.6 | рост |
## Источники
1. https://example.com/test
"""


def _llm(content: str = "") -> LLMResult:
    return LLMResult(content=content, requested_tier="t", served_tier="t", model="fake", cost_usd=0.001)


class FakeRouter:
    """Отвечает каноническими артефактами по имени схемы; ведёт счётчик вызовов."""

    def __init__(self, validator_verdicts: list[str] | None = None) -> None:
        self.calls: Counter[str] = Counter()
        self.ledger = SpendLedger()
        self.verdicts = list(validator_verdicts or ["ok"])

    def structured(self, tier: str, messages, schema: type[BaseModel], **_: Any) -> StructuredResult:
        self.calls[schema.__name__] += 1
        return StructuredResult[schema](value=self._value(schema), calls=[_llm()], first_try_valid=True)

    def chat(self, tier: str, messages, **_: Any) -> LLMResult:
        self.calls["chat"] += 1
        return _llm(REPORT)

    def _value(self, schema: type[BaseModel]) -> BaseModel:
        name = schema.__name__
        if name == "Plan":
            return Plan(ticker="TEST", tickers=["TEST"], subtasks=[SubTask(assignee="collector", goal="цены")])
        if name == "EvidenceBatch":
            return EvidenceBatch(evidence=[])
        if name == "AnalystStep":
            prediction = DraftPrediction(ticker="TEST", direction="up", horizon_days=30, confidence=0.6, thesis="рост")
            return AnalystStep(draft=AnalysisDraft(ticker="TEST", theses=["тезис"], predictions=[prediction]))
        if name == "DebateCase":
            return DebateCase(side="bull", arguments=["a", "b", "c"], weakest_point="w", summary="s")
        if name == "RiskReport":
            return RiskReport(summary="ok")
        if name == "ValidationResult":
            verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            return ValidationResult(verdict=verdict)
        if name.endswith("_args"):
            return schema(**({"ticker": "TEST"} if "ticker" in schema.model_fields else {}))
        raise AssertionError(f"unexpected schema {name}")


def _decision(gate: str, pick: str, options: list[str]) -> Decision:
    return Decision(kind="choice", distribution={o: float(o == pick) for o in options}, pick=pick, confidence=1.0,
                    latency_ms=1, gate=gate)


class FakeGates:
    """Детерминированные решения G1–G5."""

    def __init__(self, scenario: str = "analyze_ticker", risk: int = 3) -> None:
        self.scenario = scenario
        self.risk = risk
        self._tool_calls = 0

    def route_query(self, query: str) -> Decision:
        return _decision("G1", self.scenario, ["analyze_ticker", "portfolio_qa", "reject"])

    def choose_tool(self, recent_steps: str, tools: dict[str, str]) -> Decision:
        self._tool_calls += 1
        pick = "get_market_data" if self._tool_calls == 1 else "finish"
        return _decision("G2", pick, list(tools))

    def should_retry(self, error: str, context: str) -> Decision:
        return _decision("G3", "yes", ["yes", "no"])

    def decide(self, spec, gate: str = "") -> Decision:
        return _decision(gate, spec.options[0], spec.options)

    def needs_validation(self, draft: str) -> Decision:
        return _decision("G4", "yes", ["yes", "no"])

    def risk_score(self, analysis: str) -> Decision:
        return _decision("G5", str(self.risk), [str(i) for i in range(11)])


class FakeEmbedder:
    """Детерминированный эмбеддер на хэшах токенов (без загрузки моделей)."""

    dim = 64

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def dense(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                vec[self._bucket(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def sparse(self, texts: list[str]) -> list[models.SparseVector]:
        return [self.sparse_query(t) for t in texts]

    def sparse_query(self, text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for tok in self._tokens(text):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 100000
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        q = set(self._tokens(query))
        return [4.0 * len(q & set(self._tokens(d))) / (len(q) or 1) - 2.0 for d in documents]


def memory_knowledge():
    """L3 на in-memory Qdrant с детерминированным эмбеддером."""
    from reflexdesk.memory import Knowledge
    from reflexdesk.memory.knowledge import DocumentMemory, LessonMemory, ProfileMemory
    from reflexdesk.memory.semantic import SemanticStore

    store = SemanticStore(QdrantClient(location=":memory:"), FakeEmbedder())
    return Knowledge(DocumentMemory(store), ProfileMemory(store), LessonMemory(store))
