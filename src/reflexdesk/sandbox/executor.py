"""Запуск кода в отдельном процессе: политика, таймаут, лимиты; fork из прогретого родителя на Linux."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from reflexdesk.sandbox import child
from reflexdesk.sandbox.policy import PolicyViolation, check

CHILD = Path(child.__file__)
DEFAULT_TIMEOUT_S = 60
DEFAULT_MEMORY_MB = 1024


@dataclass
class ExecResult:
    """Результат исполнения: stdout, значение переменной result, ошибка."""

    stdout: str = ""
    result: object = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Runner(Protocol):
    def run(self, request: dict, timeout_s: int) -> ExecResult: ...


class SubprocessRunner:
    """Переносимый режим: новый интерпретатор -I на каждый запуск (холодный импорт библиотек)."""

    def run(self, request: dict, timeout_s: int) -> ExecResult:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(CHILD)],
                input=json.dumps(request), capture_output=True, text=True,
                timeout=timeout_s + 30, cwd=tempfile.gettempdir(), env=_minimal_env(),
            )
        except subprocess.TimeoutExpired:
            return ExecResult(error=f"TimeoutError: execution exceeded {timeout_s}s")
        try:
            return ExecResult(**json.loads(proc.stdout))
        except json.JSONDecodeError:
            return ExecResult(stdout=proc.stdout[-4000:], error=f"ChildCrashed: exit {proc.returncode}: {proc.stderr[-2000:]}")


class ForkRunner:
    """Linux: fork прогретого процесса; ребёнок ставит лимиты и хук, родитель ждёт с таймаутом."""

    def __init__(self) -> None:
        child.preload()

    def run(self, request: dict, timeout_s: int) -> ExecResult:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.close(read_fd)
                pipe = os.fdopen(write_fd, "wb")
                try:
                    payload = json.dumps(child.run(request)).encode()
                    code = 0
                except BaseException as exc:
                    payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                pipe.write(payload[: child.OUTPUT_LIMIT * 2])
                pipe.flush()
            finally:
                os._exit(code)
        os.close(write_fd)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            time.sleep(0.02)
        else:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(read_fd)
            return ExecResult(error=f"TimeoutError: execution exceeded {timeout_s}s")
        with os.fdopen(read_fd, "rb") as pipe:
            raw = pipe.read()
        try:
            return ExecResult(**json.loads(raw or b"{}"))
        except json.JSONDecodeError:
            return ExecResult(error="ChildCrashed: malformed output")


def default_runner() -> Runner:
    """ForkRunner там, где есть fork, иначе SubprocessRunner."""
    return ForkRunner() if hasattr(os, "fork") else SubprocessRunner()


def execute(
    code: str,
    runner: Runner,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> ExecResult:
    """Проверяет политику и исполняет код выбранным раннером."""
    try:
        check(code)
    except PolicyViolation as exc:
        return ExecResult(error=f"PolicyViolation: {exc}")
    return runner.run({"code": code, "memory_mb": memory_mb, "cpu_s": timeout_s}, timeout_s)


def _minimal_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return env
