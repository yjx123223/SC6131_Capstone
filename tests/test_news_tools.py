"""
tests/test_news_tools.py
-------------------------
tools.news_tools.query_news：注入假的 yfinance Ticker，覆盖新旧两种
新闻格式、新鲜度过滤、去重、排序与各类失败分支。
"""

from datetime import datetime, timezone

import pytest

from tools import news_tools

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _new_fmt(title, pub, provider="Reuters", url="https://example.com/a", summary="summary"):
    return {
        "id": title,
        "content": {
            "title": title,
            "summary": summary,
            "pubDate": pub,
            "provider": {"displayName": provider},
            "canonicalUrl": {"url": url},
        },
    }


def _old_fmt(title, epoch, publisher="Bloomberg", link="https://example.com/b"):
    return {"uuid": title, "title": title, "publisher": publisher, "link": link, "providerPublishTime": epoch}


class FakeTicker:
    def __init__(self, news=None, exc=None):
        self._news = news
        self._exc = exc
        self.count = None

    def get_news(self, count=10, tab="news"):
        self.count = count
        if self._exc:
            raise self._exc
        return self._news


def _factory(fake):
    return lambda symbol: fake


def test_parses_new_format_and_sorts_desc():
    fake = FakeTicker([
        _new_fmt("Older story", "2026-09-10T08:00:00Z"),
        _new_fmt("Newest story", "2026-09-15T21:30:00Z", provider="CNBC"),
    ])
    result = news_tools.query_news("Apple Inc.", ticker_factory=_factory(fake), now=NOW)

    assert result["ticker"] == "AAPL"
    assert result["article_count"] == 2
    first = result["articles"][0]
    assert first["title"] == "Newest story"
    assert first["publisher"] == "CNBC"
    assert first["published_at"].startswith("2026-09-15T21:30")
    assert "_published" not in first          # 内部字段不应泄露


def test_parses_old_format():
    epoch = int(datetime(2026, 9, 14, tzinfo=timezone.utc).timestamp())
    fake = FakeTicker([_old_fmt("Legacy story", epoch)])
    result = news_tools.query_news("AAPL", ticker_factory=_factory(fake), now=NOW)

    art = result["articles"][0]
    assert art["publisher"] == "Bloomberg"
    assert art["url"] == "https://example.com/b"


def test_filters_out_articles_older_than_lookback():
    fake = FakeTicker([
        _new_fmt("Fresh", "2026-09-12T00:00:00Z"),
        _new_fmt("Too old", "2026-07-01T00:00:00Z"),
    ])
    result = news_tools.query_news("AAPL", lookback_days=14, ticker_factory=_factory(fake), now=NOW)
    assert [a["title"] for a in result["articles"]] == ["Fresh"]


def test_all_articles_stale_returns_error():
    fake = FakeTicker([_new_fmt("Too old", "2025-01-01T00:00:00Z")])
    result = news_tools.query_news("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert set(result) == {"error"}


def test_dedupes_same_title_and_drops_malformed():
    fake = FakeTicker([
        _new_fmt("Same headline", "2026-09-15T10:00:00Z"),
        _new_fmt("same headline", "2026-09-15T09:00:00Z"),
        _new_fmt("No date", None),
        {"content": {"pubDate": "2026-09-15T10:00:00Z"}},     # 无标题
        "not-a-dict",
    ])
    result = news_tools.query_news("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert result["article_count"] == 1


def test_respects_max_items_and_requests_extra_for_filtering():
    fake = FakeTicker([_new_fmt(f"Story {i}", f"2026-09-15T{i:02d}:00:00Z") for i in range(10)])
    result = news_tools.query_news("AAPL", max_items=3, ticker_factory=_factory(fake), now=NOW)

    assert fake.count == 6
    assert [a["title"] for a in result["articles"]] == ["Story 9", "Story 8", "Story 7"]


def test_summary_is_truncated():
    fake = FakeTicker([_new_fmt("Long", "2026-09-15T10:00:00Z", summary="x" * 1000)])
    result = news_tools.query_news("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert len(result["articles"][0]["summary"]) == 300


@pytest.mark.parametrize("news", [[], None])
def test_no_news_returns_error_not_mock(news):
    result = news_tools.query_news("AAPL", ticker_factory=_factory(FakeTicker(news)), now=NOW)
    assert set(result) == {"error"}


def test_fetch_exception_returns_error():
    fake = FakeTicker(exc=ConnectionError("403"))
    result = news_tools.query_news("AAPL", ticker_factory=_factory(fake), now=NOW)
    assert "error" in result


def test_unknown_entity_returns_error():
    result = news_tools.query_news("Unknown Startup Corp", ticker_factory=_factory(FakeTicker([])), now=NOW)
    assert "error" in result
