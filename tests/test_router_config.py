"""AC-5: валидатор и аналитик — разные модели/режимы; инварианты конфига роутера."""

from __future__ import annotations

import copy

import pytest
import yaml

from reflexdesk.config import get_settings
from reflexdesk.router.config import RouterConfig, load_router_config


@pytest.fixture
def raw_config() -> dict:
    return yaml.safe_load(get_settings().router_config.read_text(encoding="utf-8"))


def _validated(raw: dict) -> RouterConfig:
    from reflexdesk.router.config import _interpolate

    return RouterConfig.model_validate(_interpolate(raw))


def test_shipped_config_is_valid():
    config = load_router_config(get_settings().router_config)
    assert config.tiers["tier2"].model != config.tiers["tier3"].model
    assert config.tiers["tier2"].family != config.tiers["tier3"].family
    assert config.tiers["tier2"].mode == "instruct"
    assert config.tiers["tier3"].mode == "reasoning"


def test_same_model_for_producer_and_validator_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["tiers"]["tier3"]["model"] = raw["tiers"]["tier2"]["model"]
    with pytest.raises(ValueError, match="different models"):
        _validated(raw)


def test_same_family_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["tiers"]["tier3"]["family"] = raw["tiers"]["tier2"]["family"]
    with pytest.raises(ValueError, match="different model family"):
        _validated(raw)


def test_same_mode_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["tiers"]["tier3"]["mode"] = "instruct"
    with pytest.raises(ValueError, match="reasoning mode"):
        _validated(raw)


def test_fallback_cycle_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["tiers"]["tier1"]["fallback"] = "tier2"
    with pytest.raises(ValueError, match="cycle"):
        _validated(raw)


def test_env_interpolation(raw_config, monkeypatch):
    monkeypatch.setenv("JEV_MODEL", "custom-jev")
    assert _validated(copy.deepcopy(raw_config)).tiers["gateway"].model == "custom-jev"
    monkeypatch.delenv("JEV_MODEL")
    assert _validated(copy.deepcopy(raw_config)).tiers["gateway"].model == "tacit-4b"


def test_fallback_table(raw_config):
    tiers = _validated(copy.deepcopy(raw_config)).tiers
    assert tiers["tier2"].fallback == "tier1"
    assert tiers["tier3"].fallback == "tier2"
    assert tiers["tier3"].fallback_flag == "degraded_validation"
    assert tiers["tier1"].fallback is None
