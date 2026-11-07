"""Фасад трейсинга Langfuse; выключается только явно (TRACING_ENABLED=false)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from reflexdesk.config import get_settings
from reflexdesk.obs.context import current_run_id


class TracingConfigError(RuntimeError):
    """Трейсинг включён, но Langfuse не сконфигурирован."""


class NullObservation:
    """Заглушка observation при выключенном трейсинге."""

    def update(self, **_: Any) -> None:
        return None

    def score(self, **_: Any) -> None:
        return None


@lru_cache(maxsize=1)
def _client():
    settings = get_settings()
    if not settings.tracing_enabled:
        return None
    if not settings.langfuse_configured:
        raise TracingConfigError("TRACING_ENABLED but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST are not set")
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


@contextmanager
def observe(
    name: str,
    as_type: str = "span",
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> Iterator[Any]:
    """Открывает observation в трейсе текущего прогона."""
    client = _client()
    if client is None:
        yield NullObservation()
        return
    run_id = current_run_id.get()
    trace_context = {"trace_id": client.create_trace_id(seed=run_id)} if run_id else None
    kwargs: dict[str, Any] = {"name": name, "as_type": as_type, "input": input, "metadata": metadata}
    if model:
        kwargs["model"] = model
    with client.start_as_current_observation(trace_context=trace_context, **kwargs) as obs:
        yield obs


def score_run(run_id: str, name: str, value: float | str, comment: str | None = None) -> None:
    """Пишет score в трейс прогона."""
    client = _client()
    if client is None:
        return
    client.create_score(trace_id=client.create_trace_id(seed=run_id), name=name, value=value, comment=comment)


def flush() -> None:
    """Сбрасывает накопленные события трейса."""
    client = _client()
    if client is not None:
        client.flush()


Task = Callable[[dict[str, Any]], dict[str, Any]]
Evaluate = Callable[[dict[str, Any], dict[str, Any]], dict[str, float]]


def run_experiment(dataset: str, run_name: str, items: list[dict[str, Any]], task: Task, evaluate: Evaluate) -> list[dict[str, Any]]:
    """Dataset run: в Langfuse — через run_experiment по датасету, иначе локально; возвращает результаты по кейсам."""
    results: list[dict[str, Any]] = []

    def run_item(case: dict[str, Any]) -> dict[str, Any]:
        output = task(case)
        scores = evaluate(case, output)
        results.append({"case": case, "output": output, "scores": scores})
        return {"output": output, "scores": scores}

    client = _client()
    if client is None:
        for case in items:
            run_item(case)
        return results

    from langfuse import Evaluation

    _sync_dataset(client, dataset, items)
    wanted = {f"{dataset}-{case['id']}" for case in items}
    client.run_experiment(
        name=dataset,
        run_name=run_name,
        data=[item for item in client.get_dataset(dataset).items if item.id in wanted],
        task=lambda *, item, **_: run_item(item.input),
        evaluators=[lambda *, output, **_: [Evaluation(name=k, value=v) for k, v in output["scores"].items()]],
        max_concurrency=1,
    )
    client.flush()
    return results


def _sync_dataset(client, dataset: str, items: list[dict[str, Any]]) -> None:
    try:
        client.get_dataset(dataset)
    except Exception:
        client.create_dataset(name=dataset)
    for case in items:
        client.create_dataset_item(dataset_name=dataset, id=f"{dataset}-{case['id']}", input=case)


@contextmanager
def run_scope(run_id: str, tags: list[str], metadata: dict[str, Any] | None = None) -> Iterator[None]:
    """Атрибуты трейса прогона: session_id=run_id, теги (в т.ч. eval:*), версии."""
    client = _client()
    if client is None:
        yield
        return
    from langfuse import propagate_attributes

    with propagate_attributes(session_id=run_id, tags=tags, metadata=metadata):
        yield
