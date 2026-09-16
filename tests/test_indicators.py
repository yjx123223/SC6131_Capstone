"""
tests/test_indicators.py
-------------------------
tools.indicators 技术指标纯函数：用可手算的小样本钉住数值。
"""

import pytest

from tools import indicators as ind


def test_sma_uses_last_window_values():
    assert ind.sma([1, 2, 3, 4, 5], 3) == pytest.approx(4.0)


def test_sma_insufficient_data_returns_none():
    assert ind.sma([1, 2], 3) is None
    assert ind.sma([1, 2, 3], 0) is None


def test_rsi_wilder_hand_computed_example():
    # period=2, closes=[1,2,1,2]
    # deltas=[+1,-1,+1] → 初始 avg_gain=0.5, avg_loss=0.5
    # 平滑一步：avg_gain=(0.5*1+1)/2=0.75, avg_loss=(0.5*1+0)/2=0.25 → RS=3 → RSI=75
    assert ind.rsi_wilder([1, 2, 1, 2], period=2) == pytest.approx(75.0)


def test_rsi_all_gains_is_100():
    assert ind.rsi_wilder(list(range(1, 20)), period=14) == 100.0


def test_rsi_all_losses_is_0():
    assert ind.rsi_wilder(list(range(20, 1, -1)), period=14) == pytest.approx(0.0)


def test_rsi_flat_prices_is_50():
    assert ind.rsi_wilder([10.0] * 20, period=14) == 50.0


def test_rsi_needs_period_plus_one_closes():
    assert ind.rsi_wilder([1.0] * 14, period=14) is None
    assert ind.rsi_wilder([1.0] * 15, period=14) is not None


def test_volatility_zero_for_constant_growth():
    closes = [100 * 1.01 ** i for i in range(30)]
    assert ind.volatility(closes, 20) == pytest.approx(0.0, abs=1e-9)


def test_volatility_hand_computed_sample_std():
    # 收益率 +10%, -10%：均值 0，样本方差 = (0.01+0.01)/1 = 0.02 → std≈14.142%
    closes = [100, 110, 99]
    assert ind.volatility(closes, 2) == pytest.approx(14.1421, rel=1e-4)
    assert ind.volatility(closes, 2, annualize=True) == pytest.approx(14.1421 * 252 ** 0.5, rel=1e-4)


def test_volatility_insufficient_data():
    assert ind.volatility([1, 2, 3], 20) is None


def test_pct_change():
    assert ind.pct_change([100, 105, 110], 2) == pytest.approx(10.0)
    assert ind.pct_change([100, 110], 5) is None


def test_technical_summary_signals_uptrend():
    closes = [100 + i for i in range(60)]
    s = ind.technical_summary(closes)

    assert s["last_close"] == 159
    assert s["ma20"] == pytest.approx(149.5)
    assert s["price_vs_ma20"] == "bullish"
    assert s["price_vs_ma50"] == "bullish"
    assert s["rsi14"] == 100.0
    assert s["rsi14_signal"] == "overbought"


def test_technical_summary_short_series_marks_insufficient_data():
    s = ind.technical_summary([10, 11, 12])
    assert s["ma20"] is None
    assert s["price_vs_ma20"] == "insufficient_data"
    assert s["rsi14_signal"] == "insufficient_data"
    assert s["return_period_pct"] == pytest.approx(20.0)


def test_technical_summary_empty_series():
    s = ind.technical_summary([])
    assert s["last_close"] is None
    assert s["return_period_pct"] is None
