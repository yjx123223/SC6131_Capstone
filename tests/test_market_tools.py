"""
tests/test_market_tools.py
---------------------------
tools.market_tools.query_market_data：注入假的 yfinance Ticker，不联网。
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

import config
from tools import market_tools

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)

_INFO = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "currency": "USD",
    "marketCap": 3_500_000_000_000,
    "trailingPE": 33.456,
    "profitMargins": 0.2431,
    "returnOnEquity": 1.5,
    "dividendYield": 0.44,
    "recommendationKey": "buy",
}


def _history(last_day: str, n: int = 30, start_price: float = 200.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_day, periods=n, tz="America/New_York")
    closes = [start_price + i for i in range(n)]
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1000] * n},
        index=idx,
    )


class FakeTicker:
    def __init__(self, hist=None, info=None, hist_exc=None, info_exc=None):
        self._hist = hist
        self._info = info
        self._hist_exc = hist_exc
        self._info_exc = info_exc
        self.history_kwargs = None

    def history(self, **kwargs):
        self.history_kwargs = kwargs
        if self._hist_exc:
            raise self._hist_exc
        return self._hist

    @property
    def info(self):
        if self._info_exc:
            raise self._info_exc
        return self._info


def _factory(fake):
    calls = []

    def factory(symbol):
        calls.append(symbol)
        return fake

    factory.calls = calls
    return factory


def test_happy_path_returns_structured_fresh_data():
    fake = FakeTicker(hist=_history("2026-09-15"), info=_INFO)
    factory = _factory(fake)

    result = market_tools.query_market_data("Apple Inc.", ticker_factory=factory, now=NOW)

    assert "error" not in result
    assert factory.calls == ["AAPL"]
    assert fake.history_kwargs["period"] == config.MARKET_DEFAULT_PERIOD
    assert result["ticker"] == "AAPL"
    assert result["data_as_of"] == "2026-09-15"
    assert result["company"]["name"] == "Apple Inc."
    assert result["fundamentals"]["trailing_pe"] == 33.46
    assert result["fundamentals"]["profit_margin_pct"] == 24.31      # 小数 → 百分比
    assert result["fundamentals"]["dividend_yield_pct"] == 0.44      # 已是百分比，不再 ×100
    assert result["technicals"]["last_close"] == 229.0
    assert len(result["recent_closes"]) == config.MARKET_RECENT_CLOSES
    assert result["recent_closes"][-1] == {"date": "2026-09-15", "close": 229.0}
    assert result["warnings"] == []


def test_stale_history_is_rejected():
    fake = FakeTicker(hist=_history("2026-08-01"), info=_INFO)
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=NOW)

    assert "error" in result
    assert "过期" in result["error"]


def test_data_within_staleness_threshold_is_accepted():
    # 周五收盘、下周一上午查询（相差 3 天）应该正常返回
    fake = FakeTicker(hist=_history("2026-09-11"), info=_INFO)
    monday = datetime(2026, 9, 14, 14, 0, tzinfo=timezone.utc)
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=monday)
    assert "error" not in result


def test_empty_history_returns_error_not_mock():
    fake = FakeTicker(hist=pd.DataFrame(), info=_INFO)
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert set(result) == {"error"}


def test_history_exception_returns_error():
    fake = FakeTicker(hist_exc=ConnectionError("403 blocked"))
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert "error" in result
    assert "403 blocked" in result["error"]


def test_all_nan_closes_return_error():
    hist = _history("2026-09-15")
    hist["Close"] = float("nan")
    fake = FakeTicker(hist=hist, info=_INFO)
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert "error" in result


def test_info_failure_keeps_price_data_and_adds_warning():
    fake = FakeTicker(hist=_history("2026-09-15"), info_exc=RuntimeError("quoteSummary 401"))
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=NOW)

    assert "error" not in result
    assert result["technicals"]["last_close"] == 229.0
    assert result["company"]["name"] == "AAPL"
    assert result["fundamentals"]["trailing_pe"] is None
    assert any("quoteSummary 401" in w for w in result["warnings"])


def test_empty_info_adds_warning():
    fake = FakeTicker(hist=_history("2026-09-15"), info={})
    result = market_tools.query_market_data("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert result["warnings"] == ["公司信息/财务指标为空"]


def test_unknown_entity_never_calls_data_source():
    factory = _factory(FakeTicker())
    result = market_tools.query_market_data("Unknown Startup Corp", ticker_factory=factory, now=NOW)
    assert "error" in result
    assert factory.calls == []


@pytest.mark.parametrize("period", ["5y", "max", "1d"])
def test_period_outside_allowed_window_is_rejected(period):
    factory = _factory(FakeTicker())
    result = market_tools.query_market_data("AAPL", period=period, ticker_factory=factory, now=NOW)
    assert "error" in result
    assert factory.calls == []
