"""Структурированные JSON-логи с run_id/node."""

from __future__ import annotations

import logging
import sys

import structlog

from reflexdesk.obs.context import current_node, current_run_id

_configured = False


def _add_run_context(_, __, event_dict: dict) -> dict:
    event_dict.setdefault("run_id", current_run_id.get())
    event_dict.setdefault("node", current_node.get())
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Настраивает structlog: один JSON на строку в stdout."""
    global _configured
    if _configured:
        return
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_run_context,
            structlog.processors.EventRenamer("msg"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Логгер, привязанный к имени компонента."""
    configure_logging()
    return structlog.get_logger(component=name)
