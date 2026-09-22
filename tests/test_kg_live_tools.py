"""
tests/test_kg_live_tools.py
----------------------------
tools.kg_live_tools.query_company_graph：注入假新闻源、假抽取器和种子关系，
验证端到端的图谱构建、数据复用、并行容错与推理结果。
"""

import threading

import pytest

import config
from tools.kg_live_tools import query_company_graph

SEED = [
    {"subject": "TSM", "relation": "SUPPLIES_TO", "object": "AAPL", "note": "代工"},
    {"subject": "ASML", "relation": "SUPPLIES_TO", "object": "TSM", "note": "光刻机"},
    {"subject": "AAPL", "relation": "COMPETES_WITH", "object": "GOOGL", "note": ""},
    {"subject": "NVDA", "relation": "SUPPLIES_TO", "object": "MSFT", "note": "无关"},
]


def _article(title, day="12", url=None):
    return {"title": title, "publisher": "Reuters", "published_at": f"2026-09-{day}T08:00+00:00",
            "summary": "", "url": url or f"https://n/{title}"}


NEWS = {
    "AAPL": [_article("Apple unveils iPhone 18", "10")],
    "TSM":  [_article("TSMC capacity constrained")],
    "ASML": [_article("ASML hit by export curbs", "11")],
    "GOOGL": [_article("Pixel sales surge", "09")],
}

# 每家公司抽出的事件（按 focus ticker）
EVENTS = {
    "AAPL":  [{"article_index": 0, "event_type": "PRODUCT", "polarity": "positive", "tickers": ["AAPL"],
               "summary": "iPhone 18 发布", "confidence": 0.9, "date": "2026-09-10"}],
    "TSM":   [{"article_index": 0, "event_type": "SUPPLY_CHAIN", "polarity": "negative", "tickers": ["TSM"],
               "summary": "台积电产能受限", "confidence": 0.8, "date": "2026-09-12"}],
    "ASML":  [{"article_index": 0, "event_type": "REGULATORY", "polarity": "negative", "tickers": ["ASML", "TSM"],
               "summary": "ASML 出口受限", "confidence": 1.0, "date": "2026-09-11"}],
    "GOOGL": [{"article_index": 0, "event_type": "PRODUCT", "polarity": "positive", "tickers": ["GOOGL"],
               "summary": "Pixel 销量大增", "confidence": 0.5, "date": "2026-09-09"}],
}


class FakeExtractor:
    def __init__(self, events=EVENTS, fail=()):
        self.events, self.fail = events, set(fail)
        self.calls = {}
        self.threads = set()

    def extract(self, ticker, articles, known):
        self.calls[ticker] = {"articles": articles, "known": known}
        self.threads.add(threading.get_ident())
        if ticker in self.fail:
            return {"events": [], "dropped": {}, "error": f"抽取失败 {ticker}"}
        return {"events": self.events.get(ticker, []), "dropped": {"公司无法链接": 1}}


class FakeNews:
    def __init__(self, news=NEWS, fail=()):
        self.news, self.fail = news, set(fail)
        self.calls = []

    def __call__(self, ticker, max_items):
        self.calls.append((ticker, max_items))
        if ticker in self.fail:
            return {"error": "403"}
        return {"ticker": ticker, "articles": self.news.get(ticker, [])}


PRIOR = {
    "query_market_data": {"ticker": "AAPL", "company": {"name": "Apple Inc.", "industry": "Consumer Electronics"}},
    "query_news": {"ticker": "AAPL", "articles": NEWS["AAPL"]},
    "query_sec_filings": {"ticker": "AAPL", "filings": [
        {"form": "10-Q", "filing_date": "2026-07-31", "url": "https://sec/q"}]},
}


def _run(**kw):
    params = dict(extractor=FakeExtractor(), prior_results=PRIOR, news_fetcher=FakeNews(), seed_rows=SEED)
    params.update(kw)
    return query_company_graph("Apple Inc.", **params), params


def test_builds_graph_and_propagates_risks():
    result, p = _run()

    assert result["ticker"] == "AAPL"
    assert [(n["ticker"], n["hop"], n["roles"]) for n in result["neighbors"]] == [
        ("TSM", 1, ["供应商"]), ("GOOGL", 1, ["竞争对手"]), ("ASML", 2, ["上游供应商"]),
    ]
    risks = [(r["neighbor"], r["impact"], r["score"]) for r in result["propagated_risks"]]
    assert risks == [
        ("TSM", "供应风险", 1.0),       # ASML 事件同时影响 TSM（置信度 1.0）
        ("TSM", "供应风险", 0.8),
        ("ASML", "上游供应风险", 0.5),
        ("GOOGL", "竞争压力", 0.3),
    ]
    assert result["target_events"][0]["summary"] == "iPhone 18 发布"
    assert result["target_events"][0]["source"] == "Reuters: Apple unveils iPhone 18"
    assert result["warnings"] == []


def test_reuses_prior_data_and_fetches_only_neighbors():
    result, p = _run()
    fetched = dict(p["news_fetcher"].calls)
    assert "AAPL" not in fetched                                    # 目标新闻复用
    assert fetched == {t: config.KG_NEWS_PER_COMPANY for t in ("TSM", "GOOGL", "ASML")}

    stats = result["stats"]
    assert stats["nodes_by_type"]["Industry"] == 1
    assert stats["nodes_by_type"]["Filing"] == 1
    assert stats["edges_by_relation"]["IN_INDUSTRY"] == 1
    assert stats["edges_by_relation"]["FILED"] == 1
    nodes = {n["id"]: n for n in result["_graph"]["nodes"]}
    assert nodes["company:AAPL"]["name"] == "Apple Inc."
    assert nodes["company:AAPL"]["is_target"] is True
    assert "company:NVDA" not in nodes and "company:MSFT" not in nodes   # 无关公司被剪掉


def test_fetches_target_news_when_not_in_prior():
    prior = {k: v for k, v in PRIOR.items() if k != "query_news"}
    result, p = _run(prior_results=prior)
    assert ("AAPL", config.NEWS_MAX_ITEMS) in p["news_fetcher"].calls


def test_prior_results_for_other_ticker_or_errors_are_ignored():
    prior = {
        "query_market_data": {"ticker": "MSFT", "company": {"name": "Microsoft", "industry": "Software"}},
        "query_news": {"error": "403"},
    }
    result, p = _run(prior_results=prior)
    assert "Industry" not in result["stats"]["nodes_by_type"]
    assert any(t == "AAPL" for t, _ in p["news_fetcher"].calls)


def test_extractor_receives_known_companies():
    result, p = _run()
    known = p["extractor"].calls["TSM"]["known"]
    assert set(known) == {"AAPL", "TSM", "GOOGL", "ASML"}
    assert known["AAPL"] == "Apple Inc."


def test_extraction_stats_and_ids():
    result, _ = _run()
    ext = result["stats"]["extraction"]
    assert ext["companies_ok"] == ["AAPL", "TSM", "GOOGL", "ASML"]
    assert ext["events_extracted"] == 4
    assert ext["dropped"] == {"公司无法链接": 4}
    assert result["_event_ids"] == ["E1", "E2", "E3", "E4"]


def test_neighbor_failures_become_warnings_not_errors():
    result, _ = _run(news_fetcher=FakeNews(fail={"GOOGL"}), extractor=FakeExtractor(fail={"ASML"}))

    assert "error" not in result
    assert result["stats"]["news_failed"] == ["GOOGL"]
    assert result["stats"]["extraction"]["companies_failed"] == ["ASML"]
    assert any("GOOGL" in w for w in result["warnings"])
    assert any("抽取失败 ASML" in w for w in result["warnings"])
    assert [r["neighbor"] for r in result["propagated_risks"]] == ["TSM"]


def test_fetcher_exception_is_contained():
    def boom(ticker, n):
        raise RuntimeError("timeout")
    result, _ = _run(news_fetcher=boom)
    assert set(result["stats"]["news_failed"]) == {"TSM", "GOOGL", "ASML"}


def test_company_without_seed_relations_still_builds_own_events():
    prior = {"query_news": {"ticker": "IBM", "articles": [_article("IBM quantum")]}}
    ext = FakeExtractor(events={"IBM": [{"article_index": 0, "event_type": "PRODUCT", "polarity": "positive",
                                         "tickers": ["IBM"], "summary": "量子", "confidence": 0.7,
                                         "date": "2026-09-12"}]})
    result = query_company_graph("IBM", extractor=ext, prior_results=prior,
                                 news_fetcher=FakeNews(), seed_rows=SEED)

    assert result["neighbors"] == []
    assert result["propagated_risks"] == []
    assert result["target_events"][0]["id"] == "E1"
    assert any("种子关系中没有 IBM" in w for w in result["warnings"])


def test_mermaid_and_private_fields_present():
    result, _ = _run()
    assert result["_mermaid"].startswith("graph LR")
    assert "Apple Inc. (AAPL)" in result["_mermaid"]
    assert result["_risks"][0]["sources"][0]["url"].startswith("https://n/")
    assert result["_target_events"][0]["id"] == result["target_events"][0]["id"]


def test_llm_payload_is_bounded():
    many = {"AAPL": [{"article_index": 0, "event_type": "PRODUCT", "polarity": "positive", "tickers": ["AAPL"],
                      "summary": f"事件{i}", "confidence": 0.5, "date": f"2026-09-{10 + i % 5:02d}"}
                     for i in range(20)]}
    result, _ = _run(extractor=FakeExtractor(events=many))
    assert len(result["target_events"]) == 8
    assert len(result["_target_events"]) == 20


@pytest.mark.parametrize("entity, extractor", [
    ("Unknown Startup Corp", FakeExtractor()),
    ("AAPL", None),
])
def test_hard_errors(entity, extractor):
    result = query_company_graph(entity, extractor=extractor, news_fetcher=FakeNews(), seed_rows=SEED)
    assert set(result) == {"error"}


def test_default_seed_file_is_used(monkeypatch):
    news = FakeNews()
    result = query_company_graph("AAPL", extractor=FakeExtractor(), prior_results=PRIOR, news_fetcher=news)
    assert result["neighbors"]                 # 项目种子文件里 AAPL 有邻居
    assert "ASML" in {n["ticker"] for n in result["neighbors"]}
