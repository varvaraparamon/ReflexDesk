"""Инструменты деска; импорт модулей регистрирует тулы в реестре."""

from reflexdesk.config import get_settings
from reflexdesk.tools import (  # noqa: F401
    backtest,
    calendar,
    filings,
    fundamentals,
    macro,
    market,
    news,
    python_exec,
    report,
)
from reflexdesk.tools.base import (
    DESCRIPTIONS,
    REGISTRY,
    ToolResult,
    args_schema,
    data_cutoff,
    get_tool,
)

DATA_TOOLS = ("get_market_data", "get_fundamentals", "get_filing", "get_news", "get_market_calendar")


def collector_tools() -> tuple[str, ...]:
    """Тулы Data Collector с учётом фича-флагов."""
    extra = ("get_macro",) if get_settings().fred_enabled else ()
    return DATA_TOOLS + extra


__all__ = ["DESCRIPTIONS", "REGISTRY", "ToolResult", "args_schema", "collector_tools", "data_cutoff", "get_tool"]
