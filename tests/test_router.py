"""Роутер: ретраи, фолбэки по таблице, бюджет на прогон, structured-ретрай, кэш, учёт стоимости."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from reflexdesk.cache import TTLCache
from reflexdesk.config import get_settings
from reflexdesk.obs.context import bind
from reflexdesk.router.config import load_router_config
from reflexdesk.router.results import StructuredOutputError, TierUnavailable
from reflexdesk.router.router import LLMRouter
from reflexdesk.router.structured import extract_json, parse_structured
from reflexdesk.schemas import AnalystStep


def response(content: str, model: str = "m", tokens_in: int = 1000, tokens_out: int = 500, cost: float | None = None):
    message = SimpleNamespace(content=content, reasoning_content=None, model_extra={})
    usage = SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out, prompt_cache_hit_tokens=0,
                            model_extra={"cost": cost} if cost is not None else {})
    return SimpleNamespace(choices=[SimpleNamespace(message=message, logprobs=None)], usage=usage, model=model)


def timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "http://test"))


class FakeClient:
    """Сценарий ответов по модели: элемент — ответ или исключение."""

    def __init__(self, script: dict[str, list]) -> None:
        self.script = script
        self.calls: list[str] = []
        self.formats: list[str | None] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        model = params["model"]
        self.calls.append(model)
        self.formats.append(params.get("response_format", {}).get("type"))
        item = self.script[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_router(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_URL", raising=False)

    def factory(script: dict[str, list]) -> tuple[LLMRouter, FakeClient]:
        config = load_router_config(get_settings().router_config)
        config.retry.backoff_s = 0.0
        router = LLMRouter(config, cache=TTLCache(tmp_path / "c.sqlite", "llm"))
        client = FakeClient(script)
        monkeypatch.setattr(router, "_client", lambda tier: client)
        return router, client

    return factory


class Answer(BaseModel):
    value: int


def test_retry_then_success(make_router):
    router, client = make_router({"deepseek-flash": [timeout_error(), response('{"value": 1}')]})
    result = router.chat("tier2", [{"role": "user", "content": "x"}])
    assert result.served_tier == "tier2" and client.calls == ["deepseek-flash", "deepseek-flash"]


def test_tier3_falls_back_to_tier2_with_degraded_flag(make_router):
    errors = [timeout_error() for _ in range(3)]
    router, client = make_router({"qwen/qwen3.8-flash": errors, "deepseek-flash": [response("ok")]})
    result = router.chat("tier3", [{"role": "user", "content": "x"}])
    assert result.content == "ok"
    assert "degraded_validation" in result.flags
    assert client.calls[-1] == "deepseek-flash"


def proxy_timeout():
    request = httpx.Request("POST", "http://litellm:4000/chat/completions")
    return openai.APIStatusError("litellm.Timeout", response=httpx.Response(408, request=request), body=None)


def test_proxy_timeout_is_retried_then_falls_back(make_router):
    router, client = make_router({"qwen/qwen3.8-flash": [proxy_timeout() for _ in range(3)], "deepseek-flash": [response("ok")]})
    result = router.chat("tier3", [{"role": "user", "content": "x"}])
    assert result.served_tier == "tier2" and "degraded_validation" in result.flags
    assert client.calls.count("qwen/qwen3.8-flash") == 3


def test_tier1_has_no_fallback(make_router):
    router, _ = make_router({"qwen/qwen3.8-flash": [timeout_error() for _ in range(3)]})
    with pytest.raises(TierUnavailable):
        router.chat("tier1", [{"role": "user", "content": "x"}])


def test_budget_exhaustion_triggers_fallback(make_router):
    big = response("expensive", tokens_in=1_000_000, tokens_out=0)
    router, client = make_router({"deepseek-flash": [big], "qwen/qwen3.8-flash": [response("local")]})
    with bind(run_id="budget-run"):
        first = router.chat("tier2", [{"role": "user", "content": "a"}])
        second = router.chat("tier2", [{"role": "user", "content": "b"}])
    assert first.cost_usd == pytest.approx(0.30)
    assert second.content == "local" and second.served_tier == "tier1"
    assert router.ledger.spent("budget-run", "tier2") == pytest.approx(0.30)


def test_reported_rub_cost_converted_to_usd(make_router):
    router, _ = make_router({"qwen/qwen3.8-flash": [response("ok", cost=8.0)]})
    result = router.chat("tier3", [{"role": "user", "content": "x"}])
    assert result.cost_usd == pytest.approx(8.0 / router.config.fx_per_usd["RUB"])


def test_structured_fallback_uses_format_of_serving_tier(make_router):
    errors = [timeout_error() for _ in range(3)]
    router, client = make_router({"qwen/qwen3.8-flash": errors, "deepseek-flash": [response('{"value": 3}')]})
    result = router.structured("tier3", [{"role": "user", "content": "x"}], Answer)
    assert result.value.value == 3
    assert client.formats == ["json_schema"] * 3 + ["json_object"]


def test_structured_retry_on_schema_error(make_router):
    router, client = make_router({"deepseek-flash": [response('{"value": "x"}'), response('```json\n{"value": 7}\n```')]})
    result = router.structured("tier2", [{"role": "user", "content": "x"}], Answer)
    assert result.value.value == 7 and not result.first_try_valid and len(result.calls) == 2


def test_structured_gives_up_after_one_retry(make_router):
    router, _ = make_router({"deepseek-flash": [response("nope"), response("still nope")]})
    with pytest.raises(StructuredOutputError) as info:
        router.structured("tier2", [{"role": "user", "content": "x"}], Answer)
    assert len(info.value.calls) == 2


def test_structured_mode_follows_real_provider_behind_proxy(monkeypatch):
    monkeypatch.setenv("LITELLM_URL", "http://litellm:4000")
    config = load_router_config(get_settings().router_config)
    assert config.uses_proxy
    assert config.structured_mode("tier2") == "json_object"
    assert config.structured_mode("tier3") == "json_schema"
    assert config.request_extras("tier3")["extra_body"]["reasoning"]["max_tokens"] == 3072


def test_local_tiers_cached_api_tiers_not(make_router):
    router, client = make_router({
        "tacit-4b": [response("cached")],
        "deepseek-flash": [response("a"), response("b")],
    })
    messages = [{"role": "user", "content": "same"}]
    assert router.chat("tier0", messages).content == "cached"
    assert router.chat("tier0", messages).cached
    assert router.chat("tier2", messages).content == "a"
    assert router.chat("tier2", messages).content == "b"
    assert client.calls.count("tacit-4b") == 1


def test_parse_takes_first_of_concatenated_objects():
    text = '{"code": "print(1)"}\n\n{"draft": {"theses": []}}'
    assert parse_structured(text, AnalystStep).code == "print(1)"


def test_extract_json_strips_fence_and_prose():
    assert extract_json('Вот ответ:\n```json\n{"a": 1}\n```') == '{"a": 1}'
