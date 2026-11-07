"""Jev-like gateway: формат Tacit, readout, L0, эскалация на рассуждение, отсутствие подстановок по умолчанию."""

from __future__ import annotations

import math

import httpx
import pytest

from reflexdesk.gateway import readout
from reflexdesk.gateway.backends import parse_top_logprobs
from reflexdesk.gateway.gates import Gates, GateSpecs
from reflexdesk.gateway.gateway import Gateway, GatewayError
from reflexdesk.gateway.spec import DecisionSpec


def _logsoftmax(scores: dict[str, float]) -> dict[str, float]:
    total = math.log(sum(math.exp(v) for v in scores.values()))
    return {k: v - total for k, v in scores.items()}


class ScriptedJev:
    """Логиты меток по показанному под меткой варианту; reasoning — фиксированный ответ."""

    def __init__(self, preferences: dict[str, float], label_bias: float = 0.0, reasoning_pick: str | None = None) -> None:
        self.preferences = preferences
        self.label_bias = label_bias
        self.reasoning_pick = reasoning_pick
        self.batches: list[int] = []
        self.generated = 0

    def first_token_logprobs(self, prompts: list[str]) -> list[dict[str, float]]:
        self.batches.append(len(prompts))
        return [self._one(p) for p in prompts]

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        self.generated += 1
        return "думаю...</think>"

    def _one(self, prompt: str) -> dict[str, float]:
        if prompt.endswith("Answer:"):
            return {f" {self.reasoning_pick}": 0.0, " Z": -9.0}
        empty_state = "State:\n(empty)" in prompt
        scores = {}
        for line in prompt.splitlines():
            if len(line) > 2 and line[1:3] == ". " and line[0].isalnum():
                mark, option = line[0], line[3:].split(" — ")[0]
                bias = self.label_bias if mark in ("A", "1") else 0.0
                scores[mark] = (0.0 if empty_state else self.preferences.get(option, -5.0)) + bias
        if "Answer Yes or No." in prompt:
            scores = {"Yes": self.preferences.get("yes", 0.0), "No": self.preferences.get("no", 0.0)}
        return _logsoftmax(scores)


class DownJev:
    def first_token_logprobs(self, prompts: list[str]) -> list[dict[str, float]]:
        raise httpx.ConnectError("jev down")

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        raise httpx.ConnectError("jev down")


def gateway(jev, **kwargs) -> Gateway:
    return Gateway(jev, jev, **kwargs)


def test_tacit_prompt_format():
    spec = GateSpecs.route_query("Проанализируй NVDA")
    text = readout.decision_prompt(spec)
    assert text.startswith("State:\nRequest: Проанализируй NVDA\n\nQuestion: ")
    assert "Options:\nA. analyze_ticker — " in text and text.endswith("Answer with the letter only.")
    chat = readout.chatml(readout.DECISION_SYSTEM, text, thinking=False)
    assert chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_labels_per_kind():
    assert readout.labels(GateSpecs.needs_validation("x")) == ["Yes", "No"]
    assert readout.labels(GateSpecs.risk_score("x")) == ["1", "2", "3", "4", "5", "6"]
    assert readout.labels(GateSpecs.route_query("x")) == ["A", "B", "C"]


def test_parse_vllm_logprobs():
    vllm = {"logprobs": {"top_logprobs": [{"A": -0.1, "B": -2.0}]}}
    assert parse_top_logprobs(vllm) == {"A": -0.1, "B": -2.0}
    with pytest.raises(ValueError):
        parse_top_logprobs({"logprobs": None})


def test_one_forward_confident_decision():
    jev = ScriptedJev({"analyze_ticker": 4.0})
    decision = Gates(gateway(jev)).route_query("Проанализируй NVDA")
    assert decision.pick == "analyze_ticker" and decision.route == "one_forward" and not decision.jev_fallback
    assert jev.batches == [1] and jev.generated == 0


def test_low_confidence_escalates_to_reasoning():
    jev = ScriptedJev({"analyze_ticker": 0.1, "portfolio_qa": 0.0, "reject": 0.05}, reasoning_pick="C")
    decision = Gates(gateway(jev, threshold=0.55)).route_query("Купи AAPL")
    assert decision.route == "cot" and decision.jev_fallback and decision.pick == "reject"
    assert jev.generated == 1


def test_reasoning_gate_override():
    jev = ScriptedJev({"analyze_ticker": 5.0}, reasoning_pick="C")
    decision = Gates(gateway(jev, reasoning_gates=frozenset({"G1"}))).route_query("Напиши стих")
    assert decision.pick == "reject" and decision.route == "cot"


def test_l0_removes_label_bias_in_one_batch():
    jev = ScriptedJev({"analyze_ticker": 0.0, "portfolio_qa": 0.4, "reject": -3.0}, label_bias=1.5)
    spec = GateSpecs.route_query("x")
    biased = gateway(jev, threshold=0.0).decide(spec)
    fair = gateway(jev, threshold=0.0, readout_mode="l0").decide(spec)
    assert biased.pick == "analyze_ticker" and fair.pick == "portfolio_qa"
    assert jev.batches[-1] == 6


def test_noul_maps_by_meaning():
    jev = ScriptedJev({"yes": 3.0, "no": 0.0})
    decision = Gates(gateway(jev)).needs_validation("Выручка 10 млрд")
    assert decision.pick == "yes" and decision.distribution["yes"] > 0.9


def test_score_gate_threshold():
    jev = ScriptedJev({"8": 4.0})
    decision = Gates(gateway(jev)).risk_score("агрессивный прогноз")
    assert decision.pick == "8" and Gates.is_risky(decision)


def test_backend_down_raises_instead_of_default():
    with pytest.raises(GatewayError):
        Gates(gateway(DownJev())).needs_validation("x")


def test_spec_validation():
    assert DecisionSpec(kind="noul", context="x", question="q").options == ["yes", "no"]
    with pytest.raises(ValueError):
        DecisionSpec(kind="choice", context="x", question="q", options=["a"])
