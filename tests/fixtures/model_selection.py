"""Сравнение кандидатов T1: JSON по схемам Evidence/RiskReport, FinQA с судьёй T3, tok/s, пик VRAM.

Запуск вручную; API-кандидаты идут напрямую, локальные по одному в разовом контейнере llama.cpp,
поэтому перед запуском нужно остановить сервис jev.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from dotenv import dotenv_values
from pydantic import BaseModel, Field

from reflexdesk.agents.services import Services, default_services
from reflexdesk.agents.specs import COLLECTOR, RISK_OFFICER
from reflexdesk.config import get_settings
from reflexdesk.obs.context import bind
from reflexdesk.router.router import is_transient
from reflexdesk.router.structured import parse_structured, schema_instruction
from reflexdesk.schemas import EvidenceBatch, RiskReport
from reflexdesk.tools import get_tool

ENV = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
CONTAINER = "rd-model-candidate"
PORT = 8090
FINQA_URL = "https://datasets-server.huggingface.co/rows"
FINQA_DATASET = "dreamerdeo/finqa"
JSON_TARGET = 0.95
STRUCTURED_MAX_TOKENS = 4096
FINQA_MAX_TOKENS = 1024
API_ATTEMPTS = 3
API_RETRY_PAUSE_S = 30
REQUEST_TIMEOUT_S = 900
DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "KO", "PG", "JNJ", "WMT", "INTC")
CANDIDATES: list[dict[str, Any]] = [
    {"name": "Qwen 3.8 Flash", "base_url": "https://polza.ai/api/v1", "model": "qwen/qwen3.8-flash", "api_key_env": "QWEN_API_KEY",
     "license": "Apache-2.0", "extra_body": {"reasoning": {"enabled": False}}},
    {"name": "Qwen3-8B", "hf": "Qwen/Qwen3-8B-GGUF:Q4_K_M", "license": "Apache-2.0",
     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    {"name": "Qwen2.5-7B-Instruct", "hf": "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "license": "Apache-2.0"},
    {"name": "Llama-3.1-8B-Instruct", "hf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "license": "Llama 3.1 Community"},
    {"name": "Mistral-7B-Instruct-v0.3", "hf": "bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M", "license": "Apache-2.0"},
]


@dataclass(frozen=True)
class Candidate:
    """Кандидат T1 за OpenAI-совместимым эндпоинтом; local — модель в локальной VRAM (меряется пик памяти)."""

    name: str
    base_url: str
    model: str
    license: str
    api_key: str = "none"
    local: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    name: str
    license: str
    json_valid: float = 0.0
    finqa_score: float | None = None
    tokens_per_s: float | None = None
    peak_vram_mb: int | None = None
    truncated: int = 0
    errors: list[str] = field(default_factory=list)


class FinQAGrade(BaseModel):
    """Оценка ответа на FinQA относительно эталона."""

    score: int = Field(ge=0, le=10)


class VramSampler:
    """Фоновый опрос nvidia-smi: пик занятой памяти, МБ."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.peak = 0
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out:
                self.peak = max(self.peak, int(out.splitlines()[0]))
            self._stop.wait(self._interval)

    def __enter__(self) -> VramSampler:
        if shutil.which("nvidia-smi"):
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()


def structured_tasks(services: Services, tickers: tuple[str, ...]) -> list[tuple[str, type[BaseModel], str]]:
    """Задачи (system, schema, user): извлечение Evidence из новостей/отчётности и RiskReport по фактам."""
    tasks = []
    collector = services.agent(COLLECTOR).prompt.text
    risk = services.agent(RISK_OFFICER).prompt.text
    for ticker in tickers:
        with bind(run_id=f"model-selection-{ticker}"):
            results = [get_tool(tool)(**kwargs) for tool, kwargs in
                       (("get_news", {"query": f"{ticker} stock"}), ("get_filing", {"ticker": ticker, "form": "10-Q"}))]
            facts = get_tool("get_fundamentals")(ticker=ticker)
        for result in results:
            for doc in result.documents[:2]:
                user = f"Извлеки факты о {ticker}. source_url — URL документа.\n### {doc.title}\nURL: {doc.url}\n{doc.text[:2500]}"
                tasks.append((collector, EvidenceBatch, user))
        if facts.ok:
            lines = "\n".join(f"- {e.fact}: {e.value}" for e in facts.evidence)
            tasks.append((risk, RiskReport, f"Проверь риски по фактам {ticker}:\n{lines}"))
    return tasks


def finqa_tasks(n: int = 30) -> list[dict[str, Any]]:
    """Первые n вопросов тестового сплита FinQA (контекст, вопрос, эталонный ответ)."""
    response = httpx.get(FINQA_URL, params={"dataset": FINQA_DATASET, "config": "default", "split": "test", "offset": 0, "length": n}, timeout=60)
    response.raise_for_status()
    rows = [r["row"] for r in response.json()["rows"]]
    tasks = []
    for row in rows:
        context = " ".join(row.get("pre_text") or []) + "\n" + json.dumps(row.get("table") or [], ensure_ascii=False) + "\n" + " ".join(row.get("post_text") or [])
        tasks.append({"context": context[:6000], "question": row["question"], "answer": str(row.get("answer") or row.get("final_result"))})
    return tasks


class ModelSelection:
    """Прогон всех кандидатов по одинаковым задачам."""

    def __init__(self, services: Services, candidates: list[Candidate]) -> None:
        self._services = services
        self._candidates = candidates

    def run(self, tickers: tuple[str, ...] = DEFAULT_TICKERS, finqa_n: int = 30) -> list[CandidateResult]:
        structured = structured_tasks(self._services, tickers)
        finqa = finqa_tasks(finqa_n) if finqa_n else []
        return [self._evaluate(c, structured, finqa) for c in self._candidates]

    def _evaluate(self, candidate: Candidate, structured, finqa) -> CandidateResult:
        client = openai.OpenAI(base_url=candidate.base_url, api_key=candidate.api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S)
        result = CandidateResult(candidate.name, candidate.license)
        speeds: list[float] = []
        with VramSampler() as vram:
            valid = 0
            for system, schema, user in structured:
                content, speed = self._ask(client, candidate, f"{system}\n\n{schema_instruction(schema)}", user, result,
                                           STRUCTURED_MAX_TOKENS)
                speeds += [speed] if speed else []
                try:
                    parse_structured(content, schema)
                    valid += 1
                except ValueError:
                    pass
            result.json_valid = valid / len(structured) if structured else 0.0
            grades = []
            for task in finqa:
                answer, speed = self._ask(client, candidate, "Ответь на финансовый вопрос по контексту. Дай итоговое число.",
                                          f"{task['context']}\n\nВопрос: {task['question']}", result, FINQA_MAX_TOKENS)
                speeds += [speed] if speed else []
                grades.append(self._grade(task, answer))
            result.finqa_score = sum(grades) / len(grades) if grades else None
        result.tokens_per_s = sum(speeds) / len(speeds) if speeds else None
        result.peak_vram_mb = (vram.peak or None) if candidate.local else None
        return result

    @staticmethod
    def _ask(client: openai.OpenAI, candidate: Candidate, system: str, user: str, result: CandidateResult,
             max_tokens: int) -> tuple[str, float | None]:
        """Ответ кандидата; временные сбои провайдера повторяются, обрезка по лимиту токенов считается отдельно."""
        for attempt in range(API_ATTEMPTS):
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=candidate.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0, max_tokens=max_tokens, extra_body=candidate.extra_body or None,
                )
                break
            except openai.APIError as exc:
                if not is_transient(exc) or attempt == API_ATTEMPTS - 1:
                    result.errors.append(str(exc)[:200])
                    return "", None
                time.sleep(API_RETRY_PAUSE_S)
        elapsed = time.perf_counter() - started
        if response.choices[0].finish_reason == "length":
            result.truncated += 1
        tokens = response.usage.completion_tokens if response.usage else 0
        timings = getattr(response, "timings", None) or (response.model_extra or {}).get("timings")
        speed = timings.get("predicted_per_second") if isinstance(timings, dict) else (tokens / elapsed if elapsed else None)
        return response.choices[0].message.content or "", speed

    def _grade(self, task: dict[str, Any], answer: str) -> float:
        prompt = (
            f"Вопрос: {task['question']}\nЭталонный ответ: {task['answer']}\nОтвет модели: {answer[:2000]}\n"
            "Оцени корректность ответа модели по шкале 0–10 (10 — совпадает с эталоном с точностью до округления)."
        )
        graded = self._services.router.structured("tier3", [{"role": "user", "content": prompt}], FinQAGrade)
        return float(graded.value.score)


def to_markdown(results: list[CandidateResult]) -> str:
    """Таблица сравнения кандидатов."""
    lines = [
        "| Модель | JSON valid | Обрезано по лимиту | Ошибок запроса | FinQA (0–10) | tok/s | Пик VRAM, МБ | Лицензия |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = " ✅" if r.json_valid >= JSON_TARGET else ""
        finqa = f"{r.finqa_score:.2f}" if r.finqa_score is not None else "н/д"
        speed = f"{r.tokens_per_s:.1f}" if r.tokens_per_s else "н/д"
        lines.append(f"| {r.name} | {r.json_valid:.3f}{mark} | {r.truncated} | {len(r.errors)} | {finqa} | {speed} | {r.peak_vram_mb or 'н/д'} | {r.license} |")
    return "\n".join(lines)


def _out_dir() -> Path:
    return get_settings().data_dir / "evals" / "model_selection"


def load() -> list[CandidateResult]:
    """Уже посчитанные кандидаты — для продолжения прерванного прогона."""
    path = _out_dir() / "results.json"
    return [CandidateResult(**r) for r in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []


def save(results: list[CandidateResult]) -> str:
    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "table.md").write_text(to_markdown(results), encoding="utf-8")
    return str(out)


def start(image: str, hf: str, alias: str) -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "-p", f"{PORT}:{PORT}", "-e", "LLAMA_CACHE=/models",
         "-v", "rd-models:/models", image, "-hf", hf, "--alias", alias, "-c", "16384", "-ngl", "-1", "-fa", "on",
         "--host", "0.0.0.0", "--port", str(PORT)],
        check=True, capture_output=True,
    )
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"{alias} did not start")


def stop() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def evaluate(services, spec: dict, args) -> list:
    """Результат одного кандидата: API — по base_url, локальный — в поднятом на время контейнере."""
    if "base_url" in spec:
        candidate = Candidate(spec["name"], spec["base_url"], spec["model"], spec["license"],
                              api_key=os.environ.get(spec["api_key_env"]) or ENV[spec["api_key_env"]], local=False, extra_body=spec.get("extra_body", {}))
        return ModelSelection(services, [candidate]).run(tuple(args.tickers), args.finqa)
    alias = spec["name"].lower()
    start(args.image, spec["hf"], alias)
    try:
        candidate = Candidate(spec["name"], f"http://127.0.0.1:{PORT}/v1", alias, spec["license"], extra_body=spec.get("extra_body", {}))
        return ModelSelection(services, [candidate]).run(tuple(args.tickers), args.finqa)
    finally:
        stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="ghcr.io/ggml-org/llama.cpp:server-cuda")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--finqa", type=int, default=30)
    args = parser.parse_args()
    services = default_services()
    done = {r.name: r for r in load()}
    for spec in CANDIDATES:
        if spec["name"] not in done or done[spec["name"]].errors:
            done[spec["name"]] = evaluate(services, spec, args)[0]
            save(list(done.values()))
    print(to_markdown(list(done.values())))


if __name__ == "__main__":
    main()
