"""Исполнение недоверенного кода: лимиты ресурсов, audit-hook запретов, exec с захватом stdout."""

from __future__ import annotations

import contextlib
import io
import json
import os
import site
import sys
import sysconfig

OUTPUT_LIMIT = 64 * 1024
PRELOAD = ("math", "numpy", "pandas", "scipy", "scipy.stats", "scipy.optimize", "scipy.interpolate")

BLOCKED_PREFIXES = (
    "os.", "socket.", "subprocess.", "shutil.", "ctypes.", "urllib.", "http.", "ftplib.", "smtplib.",
    "webbrowser.", "winreg.", "sqlite3.", "code.", "pty.", "compile", "builtins.input",
)
PATH_EVENTS = frozenset({"open", "os.listdir", "os.scandir"})
READ_MODES = frozenset({"r", "rb", None})


def preload() -> None:
    """Импортирует разрешённые библиотеки и прогревает ленивые подмодули pandas."""
    for name in PRELOAD:
        with contextlib.suppress(ImportError):
            __import__(name)
    with contextlib.suppress(ImportError):
        import pandas as pd

        frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
        repr(frame.describe())
        frame.resample("W").last().pct_change().rolling(2).std().to_dict()


def _library_roots() -> tuple[str, ...]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages()}
    roots.update(p for p in sysconfig.get_paths().values() if p)
    return tuple(os.path.normcase(os.path.realpath(r)) for r in roots)


def _limit_resources(memory_mb: int, cpu_s: int) -> None:
    try:
        import resource
    except ImportError:
        return
    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))


def install_guard() -> None:
    """Audit-hook: только чтение файлов библиотек; сеть, процессы, ОС и запись запрещены."""
    roots = _library_roots()

    def allowed_path(path: object) -> bool:
        if isinstance(path, int):
            return False
        if isinstance(path, bytes):
            path = path.decode(errors="ignore")
        real = os.path.normcase(os.path.realpath(str(path)))
        return real.startswith(roots)

    def hook(event: str, args: tuple) -> None:
        if event in PATH_EVENTS:
            mode = args[1] if event == "open" and len(args) > 1 else None
            if mode in READ_MODES and allowed_path(args[0] if args else ""):
                return
            raise PermissionError(f"sandbox: {event} {args[:1]!r} is forbidden")
        if event.startswith(BLOCKED_PREFIXES):
            raise PermissionError(f"sandbox: {event} is forbidden")

    sys.addaudithook(hook)


def run(request: dict) -> dict:
    """Исполняет request['code'] в текущем процессе после установки лимитов и хука (необратимо)."""
    _limit_resources(request.get("memory_mb", 1024), request.get("cpu_s", 60))
    code = compile(request["code"], "<agent>", "exec")
    stdout = io.StringIO()
    namespace: dict = {"__name__": "__agent__"}
    install_guard()
    error = None
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = namespace.get("result")
    try:
        result = json.loads(json.dumps(result, default=str))
    except (TypeError, ValueError):
        result = repr(result)
    return {"stdout": stdout.getvalue()[:OUTPUT_LIMIT], "result": result, "error": error}


def main() -> None:
    request = json.loads(sys.stdin.read())
    preload()
    payload = run(request)
    sys.__stdout__.write(json.dumps(payload)[: OUTPUT_LIMIT * 2])


if __name__ == "__main__":
    main()
