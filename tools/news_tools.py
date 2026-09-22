"""
tools/news_tools.py
--------------------
query_news 工具的实现：通过 yfinance 获取目标公司的近期新闻。

选择 yfinance 新闻接口而不是 Tavily 的原因：免费、无需 API Key、
项目已依赖 yfinance，不增加新的外部服务。

设计取舍：
  - 这里只做"取数 + 清洗 + 新鲜度过滤"，不在工具里调用 LLM 做情感分析。
    情感判断交给 Orchestrator（Claude）阅读标题/摘要后自行完成，
    既减少一次 LLM 调用，也避免工具内部的关键词打分过于粗糙。
  - 无新闻 / 全部过期 / 拉取失败 → 返回 {"error": ...}，
    不降级为 mock 新闻（假新闻会直接污染投资建议）。

兼容两种 Yahoo 新闻返回格式：
  新格式：{"id", "content": {"title", "summary", "pubDate", "provider": {...},
           "canonicalUrl": {"url"}, "clickThroughUrl": {"url"}}}
  旧格式：{"uuid", "title", "publisher", "link", "providerPublishTime": <epoch>}
"""

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import config
from .ticker_map import resolve_ticker
from .yf_client import default_ticker_factory, to_utc, utc_now


def _parse_time(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return to_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (ValueError, OSError, OverflowError):
        return None


def _parse_article(raw: dict) -> Optional[dict]:
    """把一条 Yahoo 新闻解析成统一结构；缺标题或缺发布时间则丢弃"""
    content = raw.get("content")
    if isinstance(content, dict):
        title = content.get("title")
        summary = content.get("summary") or content.get("description") or ""
        published = _parse_time(content.get("pubDate") or content.get("displayTime"))
        publisher = (content.get("provider") or {}).get("displayName")
        url = ((content.get("canonicalUrl") or {}).get("url")
               or (content.get("clickThroughUrl") or {}).get("url"))
    else:
        title = raw.get("title")
        summary = raw.get("summary") or ""
        published = _parse_time(raw.get("providerPublishTime"))
        publisher = raw.get("publisher")
        url = raw.get("link")

    if not title or published is None:
        return None

    return {
        "title":        title.strip(),
        "publisher":    publisher or "unknown",
        "published_at": published.isoformat(timespec="minutes"),
        "summary":      (summary or "").strip()[:300],
        "url":          url or "",
        "_published":   published,   # 内部排序用，返回前删除
    }


def query_news(
    entity: str,
    max_items: int = config.NEWS_MAX_ITEMS,
    lookback_days: int = config.NEWS_LOOKBACK_DAYS,
    ticker_factory: Optional[Callable] = None,
    now: Optional[datetime] = None,
) -> dict:
    """
    获取目标公司最近 lookback_days 天内的新闻（按发布时间倒序）。

    Returns
    -------
    成功：
        {
            "ticker", "entity", "source", "lookback_days",
            "article_count": int,
            "articles": [{"title", "publisher", "published_at", "summary", "url"}, ...],
        }
    失败：{"error": str}
    """
    resolved = resolve_ticker(entity)
    if "error" in resolved:
        return resolved
    ticker = resolved["ticker"]

    factory = ticker_factory or default_ticker_factory
    now = to_utc(now) if now is not None else utc_now()
    max_items = max(1, min(int(max_items), 20))

    try:
        yf_ticker = factory(ticker)
        raw_news = yf_ticker.get_news(count=max_items * 2)
    except Exception as e:
        return {"error": f"新闻拉取失败（{ticker}）：{e}"}

    if not raw_news:
        return {"error": f"未获取到 {ticker} 的新闻"}

    cutoff = now - timedelta(days=lookback_days)
    seen_titles = set()
    articles = []
    for raw in raw_news:
        if not isinstance(raw, dict):
            continue
        art = _parse_article(raw)
        if art is None or art["_published"] < cutoff or art["_published"] > now + timedelta(days=1):
            continue
        key = art["title"].lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        articles.append(art)

    if not articles:
        return {"error": f"{ticker} 最近 {lookback_days} 天内没有可用新闻"}

    articles.sort(key=lambda a: a["_published"], reverse=True)
    articles = articles[:max_items]
    for a in articles:
        del a["_published"]

    return {
        "ticker":        ticker,
        "entity":        resolved["entity"],
        "source":        "Yahoo Finance News (yfinance)",
        "lookback_days": lookback_days,
        "article_count": len(articles),
        "articles":      articles,
    }
