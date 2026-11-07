"""Jev-like gateway на Tacit: распределение по логитам (one_forward или L0); ниже порога — рассуждение той же модели (T0)."""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Literal

import httpx

from reflexdesk.config import get_settings
from reflexdesk.gateway import readout
from reflexdesk.gateway.backends import CompletionsBackend, JevBackend
from reflexdesk.gateway.spec import Decision, DecisionSpec
from reflexdesk.obs import metrics
from reflexdesk.obs.logging import get_logger
from reflexdesk.obs.tracing import observe
from reflexdesk.router.config import RouterConfig
from reflexdesk.router.router import get_router

log = get_logger("gateway")

REASONING_MAX_TOKENS = 256
L0_MAX_ROTATIONS = 4
Readout = Literal["one_forward", "l0"]


class GatewayError(RuntimeError):
    """Jev-модель не дала распределения по вариантам."""


class Gateway:
    """DecisionSpec -> Decision; маршрут one_forward или cot (эскалация на рассуждение)."""

    def __init__(
        self,
        decider: JevBackend,
        reasoner: JevBackend,
        *,
        threshold: float = 0.55,
        reasoning_gates: frozenset[str] = frozenset(),
        reasoning_max_tokens: int = REASONING_MAX_TOKENS,
        readout_mode: Readout = "one_forward",
    ) -> None:
        self._decider = decider
        self._reasoner = reasoner
        self._threshold = threshold
        self._reasoning_gates = reasoning_gates
        self._max_tokens = reasoning_max_tokens
        self._readout = readout_mode
        self._priors: dict[tuple, dict[str, float]] = {}
        self._lock = threading.Lock()

    def decide(self, spec: DecisionSpec, gate: str = "") -> Decision:
        """Решение по спецификации с трейсингом и метриками."""
        started = time.perf_counter()
        with observe(f"gate:{gate or spec.kind}", input=spec.model_dump()) as span:
            try:
                decision = self._decide(spec, gate)
            except (ValueError, httpx.HTTPError) as exc:
                raise GatewayError(f"{gate or spec.kind}: {type(exc).__name__}: {exc}") from exc
            decision.latency_ms = int((time.perf_counter() - started) * 1000)
            decision.spec = spec
            span.update(output=decision.model_dump(exclude={"spec"}), metadata={"jev_fallback": decision.jev_fallback})
        self._record(decision)
        return decision

    def _decide(self, spec: DecisionSpec, gate: str) -> Decision:
        if gate in self._reasoning_gates:
            return _build(spec, gate, self._reason(spec), route="cot")
        first = self._l0(spec) if self._readout == "l0" else self._one_forward(spec)
        if max(first.values()) >= self._threshold:
            return _build(spec, gate, first, route="one_forward")
        return _build(spec, gate, self._reason(spec), route="cot")

    def _one_forward(self, spec: DecisionSpec) -> dict[str, float]:
        """Один проход в исходном порядке вариантов — режим, на котором обучен Tacit."""
        top = self._decider.first_token_logprobs([_decision_text(spec, spec.options)])[0]
        return readout.to_options(readout.label_distribution(top, spec), spec, spec.options)

    def _l0(self, spec: DecisionSpec) -> dict[str, float]:
        """L0 (AnyJev): ротации вариантов и деление на prior меток при пустом состоянии — одним батчем."""
        orders = readout.rotations(spec, L0_MAX_ROTATIONS)
        missing = [o for o in orders if _prior_key(spec, o) not in self._priors]
        prompts = [_decision_text(spec, o) for o in orders] + [_decision_text(spec, o, state="") for o in missing]
        tops = self._decider.first_token_logprobs(prompts)
        with self._lock:
            for order, top in zip(missing, tops[len(orders):], strict=True):
                self._priors[_prior_key(spec, order)] = readout.label_distribution(top, spec)
        per_order = []
        for order, top in zip(orders, tops[: len(orders)], strict=True):
            calibrated = readout.divide_prior(readout.label_distribution(top, spec), self._priors[_prior_key(spec, order)])
            per_order.append(readout.to_options(calibrated, spec, order))
        return readout.average(per_order)

    def _reason(self, spec: DecisionSpec) -> dict[str, float]:
        """Рассуждение до </think>, затем чтение метки после «Answer:» — вероятности сохраняются."""
        prompt = readout.chatml(readout.REASONING_SYSTEM, readout.reasoning_prompt(spec), thinking=True)
        thought = readout.close_thought(self._reasoner.generate(prompt, self._max_tokens, stop=["</think>"]))
        top = self._reasoner.first_token_logprobs([prompt + thought + "\n\nAnswer:"])[0]
        return readout.to_options(readout.label_distribution(top, spec), spec, spec.options)

    @staticmethod
    def _record(decision: Decision) -> None:
        gate = decision.gate or decision.kind
        metrics.gateway_decisions_total.labels(gate, decision.pick).inc()
        metrics.gateway_latency_seconds.labels(gate).observe(decision.latency_ms / 1000)
        if decision.jev_fallback:
            metrics.jev_fallback_total.labels(gate).inc()
        log.info(
            "gateway decision", gate=gate, pick=decision.pick, confidence=round(decision.confidence, 3),
            route=decision.route, latency_ms=decision.latency_ms,
        )


def _build(spec: DecisionSpec, gate: str, distribution: dict[str, float], route: str) -> Decision:
    pick = max(distribution, key=distribution.get)
    return Decision(
        kind=spec.kind, distribution=distribution, pick=pick, confidence=distribution[pick], latency_ms=0,
        gate=gate, route=route, jev_fallback=route == "cot",
    )


def _decision_text(spec: DecisionSpec, order: list[str], state: str | None = None) -> str:
    return readout.chatml(readout.DECISION_SYSTEM, readout.decision_prompt(spec, order, state), thinking=False)


def _prior_key(spec: DecisionSpec, order: list[str]) -> tuple:
    return spec.kind, spec.question, tuple(sorted(spec.criteria.items())), tuple(order)


def backend_for(config: RouterConfig, tier: str) -> CompletionsBackend:
    """Completions-бэкенд тира gateway/tier0 из конфига роутера."""
    cfg = config.tiers[tier]
    provider = config.providers[cfg.provider]
    return CompletionsBackend(provider.base_url, cfg.model, timeout_s=cfg.timeout_s, api_key=provider.api_key() or None)


@lru_cache(maxsize=1)
def get_gateway() -> Gateway:
    """Синглтон gateway: решения — тир gateway, рассуждения — тир tier0 (тот же инстанс Tacit)."""
    settings = get_settings()
    config = get_router().config
    reasoning = frozenset(g.strip() for g in settings.gateway_reasoning_gates.split(",") if g.strip())
    return Gateway(
        backend_for(config, "gateway"), backend_for(config, "tier0"),
        threshold=settings.gateway_threshold, reasoning_gates=reasoning, readout_mode=settings.gateway_readout,
    )
