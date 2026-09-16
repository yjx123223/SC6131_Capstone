"""
tools/yf_client.py
-------------------
yfinance 的薄封装：集中处理"懒加载 import"和"当前时间"两件事，
让 market_tools / news_tools 可以通过参数注入替身，测试时不联网。
"""

from datetime import datetime, timezone


def default_ticker_factory(symbol: str):
    """返回 yfinance.Ticker 实例（首次调用时才 import yfinance）"""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("缺少 yfinance 库。请运行：pip install yfinance") from e
    return yf.Ticker(symbol)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(dt) -> datetime:
    """把 naive / 带时区的 datetime（含 pandas.Timestamp）统一成 UTC aware datetime"""
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
