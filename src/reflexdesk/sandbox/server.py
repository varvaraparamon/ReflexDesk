"""Сервер sandbox на unix-сокете: JSON-строка запроса -> JSON-строка ответа."""

from __future__ import annotations

import argparse
import json
import os
import socketserver

from reflexdesk.sandbox.executor import (
    DEFAULT_TIMEOUT_S,
    ExecResult,
    Runner,
    default_runner,
    execute,
)

MAX_REQUEST_BYTES = 256 * 1024


class SandboxHandler(socketserver.StreamRequestHandler):
    """Обрабатывает один запрос исполнения."""

    runner: Runner

    def handle(self) -> None:
        line = self.rfile.readline(MAX_REQUEST_BYTES)
        try:
            request = json.loads(line)
            timeout_s = min(int(request.get("timeout_s", DEFAULT_TIMEOUT_S)), DEFAULT_TIMEOUT_S)
            result = execute(request["code"], self.runner, timeout_s=timeout_s)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            result = ExecResult(error=f"BadRequest: {exc}")
        self.wfile.write(json.dumps(result.to_dict()).encode() + b"\n")


def serve(socket_path: str) -> None:
    """Запускает последовательный сервер на unix-сокете."""
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    SandboxHandler.runner = default_runner()
    with socketserver.UnixStreamServer(socket_path, SandboxHandler) as server:
        os.chmod(socket_path, 0o666)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/run/sandbox/sandbox.sock")
    serve(parser.parse_args().socket)
