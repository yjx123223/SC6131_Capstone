"""
tests/test_config.py
----------------------
config.py 中的 API Key 读取入口与各 Agent 独立的模型配置。
"""

import pytest

import config


@pytest.mark.parametrize("getter, env", [
    (config.get_anthropic_api_key, "ANTHROPIC_API_KEY"),
    (config.get_fred_api_key, "FRED_API_KEY"),
    (config.get_sec_user_agent, "SEC_EDGAR_USER_AGENT"),
])
def test_explicit_value_wins_then_env_then_none(monkeypatch, getter, env):
    monkeypatch.setenv(env, "from-env")
    assert getter("explicit") == "explicit"
    assert getter(None) == "from-env"
    monkeypatch.delenv(env)
    assert getter(None) is None


def test_per_agent_models_are_independent_config_entries():
    for name in ("ORCHESTRATOR_MODEL", "CRITIC_MODEL", "KG_EXTRACT_MODEL"):
        assert isinstance(getattr(config, name), str)


def test_kg_seed_file_exists():
    assert config.KG_SEED_PATH.exists()
