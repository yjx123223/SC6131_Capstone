"""
tests/test_macro_tools.py
---------------------------
tools.macro_tools.query_macro 的行为测试。

macro_agent._fetch_fred_indicators 会真的打网络请求 FRED，测试里
monkeypatch 掉它，不依赖真实网络/API Key。
"""

import macro_agent
from tools import macro_tools


_FAKE_INDICATORS = {
    "fed_funds_rate": {"value": 5.25, "date": "2024-06-01", "label": "联邦基金利率 (%)"},
    "vix":            {"value": 13.2,  "date": "2024-06-01", "label": "VIX 恐慌指数"},
}


def test_query_macro_missing_key_returns_error():
    result = macro_tools.query_macro(fred_api_key=None)
    assert "error" in result


def test_query_macro_returns_all_indicators_by_default(monkeypatch):
    monkeypatch.setattr(macro_agent, "_fetch_fred_indicators", lambda key: _FAKE_INDICATORS)

    result = macro_tools.query_macro(fred_api_key="fake-key")

    assert set(result["indicators"].keys()) == {"fed_funds_rate", "vix"}
    assert "summary_text" in result
    assert result["indicator_docs"]["vix"]


def test_query_macro_filters_requested_indicators(monkeypatch):
    monkeypatch.setattr(macro_agent, "_fetch_fred_indicators", lambda key: _FAKE_INDICATORS)

    result = macro_tools.query_macro(fred_api_key="fake-key", indicators=["vix"])

    assert list(result["indicators"].keys()) == ["vix"]
    assert list(result["indicator_docs"].keys()) == ["vix"]


def test_query_macro_fetch_failure_returns_error(monkeypatch):
    def _boom(key):
        raise RuntimeError("FRED 超时")

    monkeypatch.setattr(macro_agent, "_fetch_fred_indicators", _boom)

    result = macro_tools.query_macro(fred_api_key="fake-key")
    assert "error" in result
