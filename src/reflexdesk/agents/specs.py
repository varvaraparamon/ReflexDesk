"""Реестр агентов деска (тир, скиллы, тулы) по таблице узлов графа."""

from __future__ import annotations

from reflexdesk.agents.base import AgentSpec
from reflexdesk.tools import DATA_TOOLS

ORCHESTRATOR = AgentSpec("orchestrator", "tier2")
COLLECTOR = AgentSpec("collector", "tier1", skills=("source_logging",), tools=DATA_TOOLS)
ANALYST = AgentSpec("analyst", "tier2", skills=("ratio_analysis", "dcf_valuation", "earnings_digest"), tools=("python_exec",))
BULL = AgentSpec("bull", "tier2", skills=("debate_format",))
BEAR = AgentSpec("bear", "tier2", skills=("debate_format",))
COMPOSER = AgentSpec("composer", "tier2", skills=("house_style",), documents=("HOUSE_STYLE.md",))
RISK_OFFICER = AgentSpec("risk_officer", "tier1", skills=("risk_checklist",))
VALIDATOR = AgentSpec("validator", "tier3", skills=("claim_verification",))
REFLECTOR = AgentSpec("reflector", "tier3", skills=("lesson_extraction",), tools=("get_market_data",))

ALL = (ORCHESTRATOR, COLLECTOR, ANALYST, BULL, BEAR, COMPOSER, RISK_OFFICER, VALIDATOR, REFLECTOR)
