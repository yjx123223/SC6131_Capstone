"""
tests/test_macro_tools.py
---------------------------
tools.macro_tools.query_macro 的行为测试。

_fetch_fred_indicators 会真的请求 FRED，测试里 monkeypatch 掉它，
不依赖真实网络 / API Key。
"""

import pandas as pd

from tools import macro_tools


_FAKE_INDICATORS = {
    "fed_funds_rate": {"value": 5.25, "date": "2024-06-01", "label": "联邦基金利率 (%)"},
    "vix":            {"value": 13.2,  "date": "2024-06-01", "label": "VIX 恐慌指数"},
}


def test_query_macro_missing_key_returns_error():
    result = macro_tools.query_macro(fred_api_key=None)
    assert "error" in result


def test_query_macro_returns_all_indicators_by_default(monkeypatch):
    monkeypatch.setattr(macro_tools, "_fetch_fred_indicators", lambda key: _FAKE_INDICATORS)

    result = macro_tools.query_macro(fred_api_key="fake-key")

    assert set(result["indicators"].keys()) == {"fed_funds_rate", "vix"}
    assert "summary_text" in result
    assert result["indicator_docs"]["vix"]


def test_query_macro_filters_requested_indicators(monkeypatch):
    monkeypatch.setattr(macro_tools, "_fetch_fred_indicators", lambda key: _FAKE_INDICATORS)

    result = macro_tools.query_macro(fred_api_key="fake-key", indicators=["vix"])

    assert list(result["indicators"].keys()) == ["vix"]
    assert list(result["indicator_docs"].keys()) == ["vix"]


def test_query_macro_fetch_failure_returns_error(monkeypatch):
    def _boom(key):
        raise RuntimeError("FRED 超时")

    monkeypatch.setattr(macro_tools, "_fetch_fred_indicators", _boom)

    result = macro_tools.query_macro(fred_api_key="fake-key")
    assert "error" in result


def test_format_indicators_shows_value_or_error():
    text = macro_tools._format_indicators({
        "vix": {"value": 13.2, "date": "2024-06-01", "label": "VIX 恐慌指数"},
        "unemployment": {"value": None, "date": "N/A", "label": "失业率 (%)", "error": "timeout"},
    })
    assert "VIX 恐慌指数：13.2  （2024-06-01）" in text
    assert "失业率 (%)：N/A  （timeout）" in text


def test_fetch_fred_indicators_parses_series_and_cpi_yoy(monkeypatch):
    """用假的 fredapi.Fred 验证取最新值、CPI 同比换算和单指标失败"""
    import sys
    import types

    idx = pd.date_range("2025-01-01", periods=14, freq="MS")

    class FakeFred:
        def __init__(self, api_key):
            assert api_key == "k"

        def get_series(self, series_id):
            if series_id == "CPIAUCSL":
                return pd.Series([100.0] + [100.0] * 12 + [103.0], index=idx)   # 13 个点前为 100
            if series_id == "UNRATE":
                raise RuntimeError("timeout")
            if series_id == "VIXCLS":
                return pd.Series([float("nan")] * 14, index=idx)
            return pd.Series([4.0] * 13 + [4.33], index=idx)

    monkeypatch.setitem(sys.modules, "fredapi", types.SimpleNamespace(Fred=FakeFred))
    result = macro_tools._fetch_fred_indicators("k")

    assert result["fed_funds_rate"] == {"value": 4.33, "date": "2026-02-01", "label": "联邦基金利率 (%)"}
    assert result["cpi_yoy"]["value"] == 3.0
    assert result["unemployment"]["value"] is None and result["unemployment"]["error"] == "timeout"
    assert result["vix"] == {"value": None, "date": "N/A", "label": "VIX 恐慌指数"}
