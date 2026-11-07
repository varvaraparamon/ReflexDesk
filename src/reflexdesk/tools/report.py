"""Тул generate_report: Markdown -> md и pdf в каталоге отчётов."""

from __future__ import annotations

from reflexdesk.config import get_settings
from reflexdesk.obs.context import current_run_id
from reflexdesk.reporting.render import ensure_disclaimer, render
from reflexdesk.tools.base import ToolResult, tool


@tool("generate_report", "сохранение отчёта в Markdown и PDF")
def generate_report(markdown: str) -> ToolResult:
    """Сохраняет отчёт прогона в Markdown и PDF."""
    name = current_run_id.get() or "adhoc"
    rendered = render(ensure_disclaimer(markdown), get_settings().reports_dir, name)
    return ToolResult(
        tool="generate_report",
        data={"md_path": str(rendered.md_path), "pdf_path": str(rendered.pdf_path)},
    )
