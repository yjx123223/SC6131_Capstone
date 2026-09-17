"""
tools/market_tools.py
----------------------
query_market_data 工具的实现：通过 yfinance 获取目标公司的
实时公司信息、估值/财务指标、近期行情，并计算技术指标。

与 python-version 的差异：
  - 失败时返回 {"error": ...}，不降级为 mock 数据
  - 新鲜度校验：最新一根 K 线距今超过 config.MARKET_MAX_STALENESS_DAYS
    天即视为数据过期，返回 error
  - 公司信息（info）拉取失败不影响行情结果，记录在 warnings 字段里，
    让 LLM 知道哪部分数据缺失，而不是悄悄留空

测试方式：通过 ticker_factory / now 参数注入替身，不联网。
"""

from datetime import datetime
from typing import Callable, Optional

import config
from . import indicators
from .ticker_map import resolve_ticker
from .yf_client import default_ticker_factory, to_utc, utc_now


def _pct(value) -> Optional[float]:
    """Yahoo 以小数返回的比率（0.25）→ 百分比（25.0）"""
    try:
        return None if value is None else round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None


def _num(value, nd: int = 2) -> Optional[float]:
    try:
        return None if value is None else round(float(value), nd)
    except (TypeError, ValueError):
        return None


def _extract_company(info: dict, ticker: str) -> dict:
    return {
        "name":      info.get("longName") or info.get("shortName") or ticker,
        "sector":    info.get("sector"),
        "industry":  info.get("industry"),
        "country":   info.get("country"),
        "currency":  info.get("currency"),
        "employees": info.get("fullTimeEmployees"),
        "summary":   (info.get("longBusinessSummary") or "")[:400],
    }


def _extract_fundamentals(info: dict) -> dict:
    return {
        "market_cap":            info.get("marketCap"),
        "trailing_pe":           _num(info.get("trailingPE")),
        "forward_pe":            _num(info.get("forwardPE")),
        "price_to_book":         _num(info.get("priceToBook")),
        # 注意：Yahoo 自 2025 年起 dividendYield 已直接以百分数返回（0.44 表示 0.44%），不再 ×100
        "dividend_yield_pct":    _num(info.get("dividendYield")),
        "profit_margin_pct":     _pct(info.get("profitMargins")),
        "revenue_growth_pct":    _pct(info.get("revenueGrowth")),
        "earnings_growth_pct":   _pct(info.get("earningsGrowth")),
        "return_on_equity_pct":  _pct(info.get("returnOnEquity")),
        "debt_to_equity":        _num(info.get("debtToEquity")),
        "current_ratio":         _num(info.get("currentRatio")),
        "total_revenue":         info.get("totalRevenue"),
        "net_income":            info.get("netIncomeToCommon"),
        "free_cash_flow":        info.get("freeCashflow"),
        "beta":                  _num(info.get("beta")),
        "fifty_two_week_high":   _num(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low":    _num(info.get("fiftyTwoWeekLow")),
        "analyst_target_mean":   _num(info.get("targetMeanPrice")),
        "analyst_recommendation": info.get("recommendationKey"),
        "analyst_count":         info.get("numberOfAnalystOpinions"),
    }


def query_market_data(
    entity: str,
    period: str = config.MARKET_DEFAULT_PERIOD,
    ticker_factory: Optional[Callable] = None,
    now: Optional[datetime] = None,
) -> dict:
    """
    获取目标公司的实时市场数据。

    Parameters
    ----------
    entity         : 公司名或 ticker，如 "Apple Inc." / "AAPL"
    period         : 行情回看窗口，取值见 config.MARKET_ALLOWED_PERIODS
    ticker_factory : symbol -> yfinance.Ticker 兼容对象（测试注入用）
    now            : 当前时间（测试注入用），默认 UTC 当前时间

    Returns
    -------
    成功：
        {
            "ticker", "entity", "source", "data_as_of", "period",
            "company": {...}, "fundamentals": {...},
            "technicals": {...}, "recent_closes": [{"date", "close"}, ...],
            "warnings": [str, ...],
        }
    失败：{"error": str}
    """
    if period not in config.MARKET_ALLOWED_PERIODS:
        return {"error": f"不支持的 period '{period}'，可选：{', '.join(config.MARKET_ALLOWED_PERIODS)}"}

    resolved = resolve_ticker(entity)
    if "error" in resolved:
        return resolved
    ticker = resolved["ticker"]

    factory = ticker_factory or default_ticker_factory
    now = to_utc(now) if now is not None else utc_now()

    try:
        yf_ticker = factory(ticker)
        hist = yf_ticker.history(period=period, auto_adjust=True)
    except Exception as e:
        return {"error": f"行情数据拉取失败（{ticker}）：{e}"}

    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        return {"error": f"未获取到 {ticker} 的行情数据（ticker 可能无效或数据源不可用）"}

    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return {"error": f"{ticker} 的行情数据全部为空值"}

    last_ts = to_utc(hist.index[-1])
    age_days = (now.date() - last_ts.date()).days
    if age_days > config.MARKET_MAX_STALENESS_DAYS:
        return {
            "error": (
                f"{ticker} 的最新行情停留在 {last_ts.date()}（{age_days} 天前），"
                f"超过 {config.MARKET_MAX_STALENESS_DAYS} 天新鲜度阈值，拒绝使用过期数据"
            )
        }

    closes = [float(c) for c in hist["Close"].tolist()]
    recent = [
        {"date": to_utc(ts).date().isoformat(), "close": round(float(c), 2)}
        for ts, c in zip(hist.index[-config.MARKET_RECENT_CLOSES:], closes[-config.MARKET_RECENT_CLOSES:])
    ]

    warnings = []
    try:
        info = yf_ticker.info or {}
    except Exception as e:
        info = {}
        warnings.append(f"公司信息/财务指标拉取失败：{e}")
    if not info and not warnings:
        warnings.append("公司信息/财务指标为空")

    return {
        "ticker":        ticker,
        "entity":        resolved["entity"],
        "source":        "Yahoo Finance (yfinance)",
        "data_as_of":    last_ts.date().isoformat(),
        "period":        period,
        "company":       _extract_company(info, ticker),
        "fundamentals":  _extract_fundamentals(info),
        "technicals":    indicators.technical_summary(closes),
        "recent_closes": recent,
        "warnings":      warnings,
    }
