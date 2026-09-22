"""
tools/indicators.py
--------------------
技术指标纯函数：输入收盘价序列（按时间升序），输出数值。

全部是确定性计算，不依赖网络 / LLM，方便单测。
数据不足时返回 None，而不是编造一个数。

与 python-version 的差异：
  - RSI 使用 Wilder 平滑（业界标准定义），而不是简单平均；
    无下跌日时 RSI=100（全无波动时 50），不再用 avg_loss=0.001 硬凑
  - 波动率使用样本标准差（ddof=1），并给出年化值（×√252）
"""

from math import sqrt
from typing import Optional, Sequence

TRADING_DAYS_PER_YEAR = 252


def sma(closes: Sequence[float], window: int) -> Optional[float]:
    """简单移动平均：最后 window 个收盘价的均值"""
    if window <= 0 or len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def rsi_wilder(closes: Sequence[float], period: int = 14) -> Optional[float]:
    """
    Wilder RSI。至少需要 period + 1 个收盘价。

    算法：
      1. 前 period 个涨跌幅的简单平均作为初始 avg_gain / avg_loss
      2. 之后每一步：avg = (avg * (period - 1) + 当前值) / period
      3. RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    """
    if period <= 0 or len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def daily_returns(closes: Sequence[float]) -> list[float]:
    """逐日简单收益率"""
    return [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]


def volatility(closes: Sequence[float], window: int = 20, annualize: bool = False) -> Optional[float]:
    """
    最近 window 个交易日收益率的样本标准差（百分比）。
    需要 window + 1 个收盘价才能得到 window 个收益率。
    """
    if window < 2 or len(closes) < window + 1:
        return None
    rets = daily_returns(closes[-(window + 1):])
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = sqrt(var) * 100
    return vol * sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def pct_change(closes: Sequence[float], lookback: int) -> Optional[float]:
    """最近 lookback 个交易日的累计涨跌幅（百分比）"""
    if lookback <= 0 or len(closes) < lookback + 1 or not closes[-(lookback + 1)]:
        return None
    return (closes[-1] / closes[-(lookback + 1)] - 1) * 100


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(x, nd)


def technical_summary(closes: Sequence[float]) -> dict:
    """
    汇总常用技术指标，并给出基于规则的信号标签（bullish / bearish / neutral）。
    数据不足的指标值为 None，信号为 "insufficient_data"。
    """
    last = closes[-1] if closes else None
    ma20 = sma(closes, 20)
    ma50 = sma(closes, 50)
    rsi = rsi_wilder(closes, 14)
    vol20 = volatility(closes, 20)
    vol20_ann = volatility(closes, 20, annualize=True)

    def _vs_ma(ma):
        if ma is None or last is None:
            return "insufficient_data"
        return "bullish" if last > ma else "bearish"

    if rsi is None:
        rsi_signal = "insufficient_data"
    elif rsi >= 70:
        rsi_signal = "overbought"
    elif rsi <= 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    return {
        "last_close":            _round(last),
        "ma20":                  _round(ma20),
        "ma50":                  _round(ma50),
        "price_vs_ma20":         _vs_ma(ma20),
        "price_vs_ma50":         _vs_ma(ma50),
        "rsi14":                 _round(rsi),
        "rsi14_signal":          rsi_signal,
        "volatility_20d_pct":    _round(vol20),
        "volatility_20d_annualized_pct": _round(vol20_ann),
        "return_5d_pct":         _round(pct_change(closes, 5)),
        "return_20d_pct":        _round(pct_change(closes, 20)),
        "return_period_pct":     _round(pct_change(closes, len(closes) - 1)) if len(closes) > 1 else None,
    }
