"""CLI деска: analyze, reflect, eval, serve, resume, predictions, feedback."""

from __future__ import annotations

import json
from datetime import date

import typer

app = typer.Typer(add_completion=False, help="ReflexDesk — мультиагентный research-деск с рефлексией")
eval_app = typer.Typer(help="Оценка качества")
app.add_typer(eval_app, name="eval")


def _print(payload) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command()
def analyze(
    query: str = typer.Argument(..., help="Запрос: тикер или вопрос"),
    as_of: str | None = typer.Option(None, "--as-of", help="Дата среза YYYY-MM-DD"),
    config: str = typer.Option("full", help="Контур: full | debate | single"),
) -> None:
    """Прогон от запроса до отчёта."""
    from reflexdesk.graph import CONFIGS
    from reflexdesk.service import DeskService

    service = DeskService(config=CONFIGS[config])
    final = service.analyze(query, date.fromisoformat(as_of) if as_of else None)
    _print(service.summary(final["run_id"]))


@app.command()
def resume(run_id: str | None = typer.Argument(None, help="run_id; без аргумента — все прерванные")) -> None:
    """Возобновление прогона с последнего чекпоинта."""
    from reflexdesk.service import DeskService

    service = DeskService()
    if run_id:
        service.execute(run_id)
        _print(service.summary(run_id))
    else:
        _print({"resumed": service.resume(("running", "interrupted"))})


@app.command()
def reflect(run_now: bool = typer.Option(False, "--run-now", help="Запустить сверку немедленно")) -> None:
    """Сверка созревших прогнозов и извлечение уроков."""
    from reflexdesk.agents.reflector import Reflector
    from reflexdesk.agents.services import default_services

    if not run_now:
        typer.echo("Reflector запускается по расписанию (scripts/reflector_cron.py); для ручного запуска: --run-now")
        raise typer.Exit(0)
    _print(Reflector(default_services()).run().__dict__)


@app.command()
def predictions(ticker: str | None = None, status: str | None = None) -> None:
    """Журнал прогнозов L2."""
    from reflexdesk.memory import get_episodic

    _print([p.model_dump(mode="json") for p in get_episodic().predictions(ticker=ticker, status=status)])


@app.command()
def feedback(run_id: str, positive: bool = typer.Option(..., "--up/--down", help="👍 / 👎")) -> None:
    """Пользовательская оценка отчёта в Langfuse."""
    from reflexdesk.obs.tracing import flush, score_run

    score_run(run_id, "user_feedback", 1.0 if positive else 0.0)
    flush()
    typer.echo("ok")


@eval_app.command("golden")
def eval_golden(
    config: str = typer.Option("full", help="Контур: full | debate | single"),
    limit: int | None = typer.Option(None, help="Ограничить число кейсов"),
) -> None:
    """Golden-set: прогоны на исторических датах, faithfulness и judge-рубрика."""
    from reflexdesk.evals.golden import run_golden

    _print(run_golden(config, limit=limit))


@eval_app.command("ablation")
def eval_ablation(limit: int | None = typer.Option(None, help="Ограничить число кейсов")) -> None:
    """Ablation: single vs debate vs full на golden-set."""
    from reflexdesk.evals.ablation import run_ablation

    _print(run_ablation(limit=limit))


@eval_app.command("backtest")
def eval_backtest(benchmark: str = typer.Option("SPY")) -> None:
    """Бэктест сверенных прогнозов против бенчмарка."""
    from reflexdesk.evals.backtest import run_backtest_report

    _print(run_backtest_report(benchmark))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """HTTP API."""
    import uvicorn

    uvicorn.run("reflexdesk.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
