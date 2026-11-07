"""Статическая проверка кода: whitelist импортов и запрет опасных конструкций."""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "scipy", "math"})
FORBIDDEN_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
})


class PolicyViolation(ValueError):
    """Код нарушает политику sandbox."""


def check(code: str) -> None:
    """Бросает PolicyViolation, если код импортирует запрещённое или трогает опасные имена."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PolicyViolation(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise PolicyViolation("relative imports are forbidden")
            _check_module(node.module or "")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise PolicyViolation(f"name {node.id!r} is forbidden")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PolicyViolation(f"dunder attribute {node.attr!r} is forbidden")


def _check_module(name: str) -> None:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise PolicyViolation(f"import of {name!r} is not allowed; allowed: {sorted(ALLOWED_IMPORTS)}")
