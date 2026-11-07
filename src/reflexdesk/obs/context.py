"""Контекстные переменные run_id и узла графа."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
current_node: ContextVar[str | None] = ContextVar("current_node", default=None)


@contextmanager
def bind(run_id: str | None = None, node: str | None = None) -> Iterator[None]:
    """Привязывает run_id и/или узел на время блока."""
    tokens = []
    if run_id is not None:
        tokens.append((current_run_id, current_run_id.set(run_id)))
    if node is not None:
        tokens.append((current_node, current_node.set(node)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
