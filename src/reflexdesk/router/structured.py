"""Промптинг по JSON Schema и разбор structured output."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema Pydantic-модели."""
    return model.model_json_schema()


def schema_instruction(model: type[BaseModel]) -> str:
    """Суффикс системного промпта с требованием JSON по схеме."""
    schema = json.dumps(schema_of(model), ensure_ascii=False)
    return (
        "Ответ — строго один JSON-объект без пояснений и без markdown, "
        f"соответствующий JSON Schema:\n{schema}"
    )


def response_format(model: type[BaseModel], mode: str) -> dict[str, Any]:
    """response_format под structured-режим тира."""
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": model.__name__, "schema": schema_of(model), "strict": False},
        }
    return {"type": "json_object"}


def extract_json(text: str) -> str:
    """Первый JSON-объект ответа (без ``` и прозы); следующие за ним объекты отбрасываются."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return cleaned
    try:
        _, stop = json.JSONDecoder().raw_decode(cleaned, start)
    except json.JSONDecodeError:
        return cleaned[start : end + 1]
    return cleaned[start:stop]


def parse_structured(text: str, model: type[T]) -> T:
    """Валидирует ответ по схеме; ValueError с читаемой причиной."""
    try:
        return model.model_validate_json(extract_json(text))
    except ValidationError as exc:
        raise ValueError(_format_errors(exc)) from exc


def _format_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()[:10]
    )
