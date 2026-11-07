"""Jev-like слой решений."""

from reflexdesk.gateway.gates import Gates
from reflexdesk.gateway.gateway import Gateway, get_gateway
from reflexdesk.gateway.spec import Decision, DecisionSpec

__all__ = ["Decision", "DecisionSpec", "Gates", "Gateway", "get_gateway"]
