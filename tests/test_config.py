"""
tests/test_config.py
----------------------
config.py 里新增的统一入口的行为测试：
  - get_anthropic_api_key / get_fred_api_key 的优先级（显式传入 > 环境变量）
  - DATA_DIR / EMBEDDING_DIR 使用同一份候选根目录逻辑
"""

import importlib
from pathlib import Path

import config


def test_get_anthropic_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert config.get_anthropic_api_key("explicit-key") == "explicit-key"


def test_get_anthropic_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert config.get_anthropic_api_key(None) == "env-key"


def test_get_anthropic_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert config.get_anthropic_api_key(None) is None


def test_get_fred_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "env-fred-key")
    assert config.get_fred_api_key("explicit-fred-key") == "explicit-fred-key"


def test_get_fred_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "env-fred-key")
    assert config.get_fred_api_key(None) == "env-fred-key"


def test_data_dir_and_embedding_dir_share_same_root_candidates():
    """
    DATA_DIR 和 EMBEDDING_DIR 都应该来自 _KG_ROOT_CANDIDATES 里的
    同一个根目录（要么都指向 knowledgeGraph/FinDKG，要么都指向同级
    FinDKG），不应该出现两者各自选中不同根目录的情况。
    """
    data_root = config.DATA_DIR.parent.parent  # .../<root>/FinDKG_dataset/FinDKG-full -> <root>
    embedding_root = config.EMBEDDING_DIR.parent  # .../<root>/embeddings -> <root>
    assert data_root == embedding_root


def test_per_agent_models_are_independent_config_entries():
    """
    各 Agent 的 model 配置应该是独立的常量（哪怕当前取值相同），
    这样以后只改 CRITIC_MODEL 不会影响 ADVISOR_MODEL / ORCHESTRATOR_MODEL。
    """
    assert isinstance(config.ADVISOR_MODEL, str)
    assert isinstance(config.ORCHESTRATOR_MODEL, str)
    assert isinstance(config.CRITIC_MODEL, str)
    assert isinstance(config.MACRO_MODEL, str)
