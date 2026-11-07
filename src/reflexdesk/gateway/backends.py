"""Бэкенд Jev-модели: OpenAI-совместимый /v1/completions (vLLM) — логиты первого токена и генерация."""

from __future__ import annotations

from typing import Protocol

import httpx

TOP_LOGPROBS = 20


class JevBackend(Protocol):
    """Чтение распределения первого токена и генерация рассуждения."""

    def first_token_logprobs(self, prompts: list[str]) -> list[dict[str, float]]: ...

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str: ...


class CompletionsBackend:
    """Сырые промпты в /v1/completions; температура 0, top-k logprobs первого токена."""

    def __init__(self, base_url: str, model: str, timeout_s: float, api_key: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._url = base_url.rstrip("/") + "/completions"
        self._model = model
        self._client = httpx.Client(timeout=timeout_s, headers=headers)

    def first_token_logprobs(self, prompts: list[str]) -> list[dict[str, float]]:
        choices = self._complete(prompt=prompts, max_tokens=1, temperature=0.0, logprobs=TOP_LOGPROBS)
        return [parse_top_logprobs(c) for c in choices]

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        return self._complete(prompt=prompt, max_tokens=max_tokens, temperature=0.0, stop=stop)[0]["text"]

    def _complete(self, **params) -> list[dict]:
        response = self._client.post(self._url, json={"model": self._model, **params})
        response.raise_for_status()
        return sorted(response.json()["choices"], key=lambda c: c.get("index", 0))


def parse_top_logprobs(choice: dict) -> dict[str, float]:
    """top-k первого токена из ответа vLLM (`top_logprobs[0]`: dict токен → logprob)."""
    logprobs = choice.get("logprobs") or {}
    if logprobs.get("top_logprobs"):
        return dict(logprobs["top_logprobs"][0])
    raise ValueError("completion has no logprobs")
