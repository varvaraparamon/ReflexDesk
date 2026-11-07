"""Тул python_exec: клиент sandbox-контейнера (unix-сокет)."""

from __future__ import annotations

import json
import socket

from reflexdesk.config import get_settings
from reflexdesk.sandbox.executor import ExecResult
from reflexdesk.tools.base import ToolResult, tool

TIMEOUT_S = 60


def _via_socket(path: str, code: str, timeout_s: int) -> ExecResult:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s + 10)
        conn.connect(path)
        conn.sendall(json.dumps({"code": code, "timeout_s": timeout_s}).encode() + b"\n")
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buffer += chunk
    return ExecResult(**json.loads(buffer))


def run_code(code: str, timeout_s: int = TIMEOUT_S) -> ExecResult:
    """Исполняет код в sandbox-контейнере через unix-сокет."""
    return _via_socket(get_settings().sandbox_socket, code, timeout_s)


@tool("python_exec", "исполнение Python (pandas, numpy, scipy, math) в изолированном sandbox")
def python_exec(code: str) -> ToolResult:
    """Код исполняется без сети и файловой системы; результат — stdout и переменная result."""
    result = run_code(code)
    return ToolResult(tool="python_exec", ok=result.error is None, data=result.to_dict(), error=result.error)
