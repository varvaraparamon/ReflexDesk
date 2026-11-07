"""Разбор Markdown-отчёта: секции и таблица прогнозов."""

from __future__ import annotations

import re

from pydantic import ValidationError

from reflexdesk.schemas import DraftPrediction

PREDICTIONS_HEADING = "Прогнозы"
SOURCES_HEADING = "Источники"
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_DIRECTIONS = {"up": "up", "down": "down", "flat": "flat", "рост": "up", "падение": "down", "боковик": "flat"}


def sections(markdown: str) -> dict[str, str]:
    """Тексты разделов второго уровня по заголовку."""
    matches = list(_HEADING.finditer(markdown))
    result = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        result[match.group(1).strip()] = markdown[match.end():end].strip()
    return result


def section(markdown: str, title: str) -> str | None:
    """Текст раздела по началу заголовка (без учёта регистра)."""
    for name, body in sections(markdown).items():
        if name.lower().startswith(title.lower()):
            return body
    return None


def table_rows(block: str) -> list[list[str]]:
    """Строки markdown-таблицы без заголовка и разделителя."""
    rows = [line.strip().strip("|").split("|") for line in block.splitlines() if line.strip().startswith("|")]
    rows = [[cell.strip() for cell in row] for row in rows]
    return [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]


def parse_predictions(markdown: str) -> tuple[list[DraftPrediction], list[str]]:
    """Прогнозы из таблицы раздела «Прогнозы» и список ошибок разбора."""
    block = section(markdown, PREDICTIONS_HEADING)
    if block is None:
        return [], ["нет раздела «## Прогнозы»"]
    predictions, errors = [], []
    for row in table_rows(block):
        if len(row) < 5:
            errors.append(f"строка прогноза неполная: {row}")
            continue
        ticker, direction, horizon, confidence, thesis = row[:5]
        try:
            predictions.append(
                DraftPrediction(
                    ticker=re.sub(r"[^A-Z.\-]", "", ticker.upper()),
                    direction=_DIRECTIONS.get(direction.strip().lower().strip("`*"), direction.strip().lower()),
                    horizon_days=int(re.sub(r"\D", "", horizon) or 0),
                    confidence=float(confidence.replace(",", ".").strip("% ")) / (100 if "%" in confidence else 1),
                    thesis=thesis,
                )
            )
        except (ValueError, ValidationError) as exc:
            errors.append(f"прогноз {row}: {exc}")
    if not predictions and not errors:
        errors.append("таблица прогнозов пуста")
    return predictions, errors
