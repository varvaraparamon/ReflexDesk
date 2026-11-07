"""LangGraph-граф деска: узлы, рёбра, конфигурации ablation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from reflexdesk.agents.analyst import AnalystNode
from reflexdesk.agents.collector import CollectorNode
from reflexdesk.agents.composer import ComposerNode
from reflexdesk.agents.debate import DebateNode
from reflexdesk.agents.finalize import FinalizeNode
from reflexdesk.agents.orchestrator import IntakeNode, OrchestratorNode
from reflexdesk.agents.risk import RiskGateNode, RiskOfficerNode
from reflexdesk.agents.services import Services
from reflexdesk.agents.validator import ValidationGateNode, ValidatorNode
from reflexdesk.gateway.gates import RISK_THRESHOLD
from reflexdesk.obs.context import bind
from reflexdesk.obs.tracing import observe
from reflexdesk.schemas import as_of_from_iso
from reflexdesk.state import DeskState
from reflexdesk.tools import data_cutoff

NodeFn = Callable[[DeskState], dict[str, Any]]
TRAJECTORY_EXCLUDE = ("evidence",)


@dataclass(frozen=True)
class GraphConfig:
    """Состав контура: дебаты и валидация (для ablation-протокола)."""

    debate: bool = True
    validation: bool = True
    name: str = "full"


FULL = GraphConfig()
WITH_DEBATE = GraphConfig(debate=True, validation=False, name="debate")
SINGLE_ANALYST = GraphConfig(debate=False, validation=False, name="single")
CONFIGS = {c.name: c for c in (FULL, WITH_DEBATE, SINGLE_ANALYST)}


def _artifact(update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in update.items():
        if key in TRAJECTORY_EXCLUDE:
            out[f"{key}_count"] = len(value)
        elif isinstance(value, BaseModel):
            out[key] = value.model_dump(mode="json")
        else:
            out[key] = value
    return out


def instrument(name: str, fn: NodeFn, services: Services) -> NodeFn:
    """Обёртка узла: run-контекст, срез данных, span, запись траектории в L2."""

    def node(state: DeskState) -> dict[str, Any]:
        run_id = state["run_id"]
        cutoff = as_of_from_iso(state["as_of_date"]) if state.get("as_of_date") else None
        with bind(run_id=run_id, node=name), data_cutoff(cutoff), observe(f"node:{name}", "agent") as span:
            update = fn(state)
            artifact = _artifact(update)
            services.episodic.record_decision(run_id, name, artifact)
            span.update(output=artifact)
        return update

    node.__name__ = name
    return node


def _after_intake(state: DeskState) -> str:
    return "finalize" if state.get("status") == "rejected" else "orchestrator"


def _after_orchestrator(state: DeskState) -> str:
    return "finalize" if state.get("status") == "failed" else "collector"


def _after_analyst(config: GraphConfig) -> Callable[[DeskState], str]:
    def route(state: DeskState) -> str:
        if state.get("analysis_draft") is None:
            return "finalize"
        return "risk_gate" if config.debate else "composer"

    return route


def _after_risk_gate(always_risk: bool) -> Callable[[DeskState], list[str]]:
    def route(state: DeskState) -> list[str]:
        if always_risk or (state.get("risk_score") or 0) >= RISK_THRESHOLD:
            return ["risk_officer"]
        return ["bull", "bear"]

    return route


def _after_validation_gate(state: DeskState) -> str:
    validation = state.get("validation")
    return "finalize" if validation is not None and validation.skipped else "validator"


def _after_validator(max_rounds: int) -> Callable[[DeskState], str]:
    def route(state: DeskState) -> str:
        validation = state.get("validation")
        passed = validation is not None and validation.verdict == "ok" and not validation.degraded
        if passed or state.get("validation_round", 0) >= max_rounds:
            return "finalize"
        return "composer"

    return route


def build_graph(services: Services, checkpointer: BaseCheckpointSaver | None = None, config: GraphConfig = FULL):
    """Компилирует граф под конфигурацию контура."""
    nodes: dict[str, NodeFn] = {
        "intake": IntakeNode(services),
        "orchestrator": OrchestratorNode(services),
        "collector": CollectorNode(services),
        "analyst": AnalystNode(services),
        "composer": ComposerNode(services),
        "finalize": FinalizeNode(services),
    }
    if config.debate:
        nodes |= {
            "risk_gate": RiskGateNode(services),
            "risk_officer": RiskOfficerNode(services),
            "bull": DebateNode(services, "bull"),
            "bear": DebateNode(services, "bear"),
        }
    if config.validation:
        nodes |= {"validation_gate": ValidationGateNode(services), "validator": ValidatorNode(services)}

    graph = StateGraph(DeskState)
    for name, fn in nodes.items():
        graph.add_node(name, instrument(name, fn, services))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _after_intake, ["orchestrator", "finalize"])
    graph.add_conditional_edges("orchestrator", _after_orchestrator, ["collector", "finalize"])
    graph.add_edge("collector", "analyst")
    graph.add_conditional_edges(
        "analyst", _after_analyst(config), ["risk_gate", "finalize"] if config.debate else ["composer", "finalize"]
    )
    if config.debate:
        graph.add_conditional_edges(
            "risk_gate", _after_risk_gate(services.settings.risk_officer_always), ["risk_officer", "bull", "bear"]
        )
        graph.add_edge("risk_officer", "bull")
        graph.add_edge("risk_officer", "bear")
        graph.add_edge(["bull", "bear"], "composer")
    if config.validation:
        graph.add_edge("composer", "validation_gate")
        graph.add_conditional_edges("validation_gate", _after_validation_gate, ["validator", "finalize"])
        graph.add_conditional_edges(
            "validator", _after_validator(services.settings.max_validation_rounds), ["composer", "finalize"]
        )
    else:
        graph.add_edge("composer", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
