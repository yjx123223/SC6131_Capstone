"""
tests/test_eval_snapshot.py
----------------------------
eval.snapshot：录制、保存/读取、回放的参数宽松匹配与隔离（不访问外部接口）。
"""

import json

import pytest

import config
from eval.snapshot import SnapshotStore, entry_key, load_companies, universe_for


def _article(t, url):
    return {"title": t, "publisher": "Reuters", "published_at": "2026-09-12T08:00+00:00",
            "summary": "", "url": url}


def _fake_fns(fail_sec=False):
    def market(ticker, period="3mo"):
        return {"ticker": ticker, "period": period, "data_as_of": "2026-09-17",
                "company": {"name": ticker, "industry": "Tech"}, "technicals": {}, "fundamentals": {},
                "warnings": []}

    def news(ticker, max_items=8):
        return {"ticker": ticker, "lookback_days": 14, "article_count": 3,
                "articles": [_article(f"{ticker} news {i}", f"https://n/{ticker}/{i}") for i in range(3)]}

    def sec(ticker, form_types=None):
        if fail_sec:
            return {"error": "未配置 SEC_EDGAR_USER_AGENT"}
        return {"ticker": ticker, "filings": [
            {"form": "10-Q", "filing_date": "2026-07-31", "url": "u1"},
            {"form": "8-K", "filing_date": "2026-08-01", "url": "u2"}]}

    def macro(key, indicators=None):
        return {"indicators": {"vix": {"value": 15.0, "date": "2026-09-16", "label": "VIX 恐慌指数"},
                               "cpi_yoy": {"value": 2.9, "date": "2026-08-01", "label": "CPI 同比变化 (%)"}},
                "indicator_docs": {"vix": "VIX 说明", "cpi_yoy": "CPI 说明"},
                "summary_text": "原始文本"}

    return dict(market_fn=market, news_fn=news, sec_fn=sec, macro_fn=macro)


@pytest.fixture
def store():
    return SnapshotStore.record(["AAPL"], on_progress=lambda m: None, **_fake_fns())


# ── 录制 ────────────────────────────────────────────────────────

def test_records_target_and_neighbors(store):
    universe = store.universe
    assert store.targets == ["AAPL"]
    assert {"AAPL", "TSM", "ASML", "GOOGL", "MSFT", "AVGO", "QCOM"} <= set(universe)
    # universe 里每家都有行情和新闻，只有目标公司有 SEC，宏观只录一次
    for t in universe:
        assert store.raw("query_market_data", t) is not None
        assert store.raw("query_news", t) is not None
    assert store.raw("query_sec_filings", "TSM") is None
    assert store.raw("query_sec_filings", "AAPL") is not None
    assert store.raw("query_macro") is not None
    assert [(n["ticker"], n["hop"]) for n in store.neighbors("AAPL")][-1] == ("ASML", 2)


def test_stats_and_failed_entries():
    store = SnapshotStore.record(["AAPL"], on_progress=lambda m: None, **_fake_fns(fail_sec=True))
    stats = store.stats()
    assert stats["targets"] == 1 and stats["universe"] == len(store.universe)
    assert stats["failed_entries"] == [entry_key("query_sec_filings", "AAPL")]
    assert stats["articles_total"] == 3 * len(store.universe)


def test_save_and_load_roundtrip(store, tmp_path):
    path = store.save(tmp_path / "snap" / "snapshot.json")
    loaded = SnapshotStore.load(path)
    assert loaded.data == store.data          # JSON 往返后结构完全一致（path 已统一为 list）
    assert loaded.neighbors("AAPL")[-1]["path"] == [["ASML", "SUPPLIES_TO", "TSM"],
                                                    ["TSM", "SUPPLIES_TO", "AAPL"]]
    assert json.loads(path.read_text(encoding="utf-8"))["targets"] == ["AAPL"]


# ── 回放 ────────────────────────────────────────────────────────

def test_replay_returns_recorded_result(store):
    assert store.replay_market("Apple Inc.")["ticker"] == "AAPL"
    assert store.replay_news("AAPL")["article_count"] == 3


def test_replay_returns_copies_not_shared_state(store):
    first = store.replay_news("AAPL")
    first["articles"].clear()
    assert len(store.replay_news("AAPL")["articles"]) == 3


def test_replay_news_truncates_to_max_items(store):
    result = store.replay_news("AAPL", max_items=2)
    assert len(result["articles"]) == 2 and result["article_count"] == 2


def test_replay_sec_filters_form_types(store):
    assert [f["form"] for f in store.replay_sec("AAPL", form_types=["8-k"])["filings"]] == ["8-K"]
    assert len(store.replay_sec("AAPL")["filings"]) == 2


def test_replay_macro_filters_indicators_and_rebuilds_summary(store):
    result = store.replay_macro(["vix"])
    assert list(result["indicators"]) == ["vix"] and list(result["indicator_docs"]) == ["vix"]
    assert "VIX 恐慌指数：15.0" in result["summary_text"]      # 按过滤后的指标重新生成
    assert list(store.replay_macro()["indicators"]) == ["vix", "cpi_yoy"]


def test_replay_unknown_ticker_does_not_hit_network(store):
    result = store.replay_news("IBM")
    assert "快照中没有 IBM" in result["error"]
    assert "error" in store.replay_market("Unknown Startup Corp")   # 无法解析 ticker


def test_restrict_to_blocks_other_companies(store):
    assert store.replay_news("TSM")["ticker"] == "TSM"
    blocked = store.replay_news("TSM", restrict_to="AAPL")
    assert "只分析目标公司 AAPL" in blocked["error"]
    assert store.replay_news("Apple Inc.", restrict_to="AAPL")["ticker"] == "AAPL"


# ── 注入 OrchestratorLoop ───────────────────────────────────────

def test_tool_impls_signature_and_coverage(store):
    impls = store.tool_impls(target="AAPL")
    assert set(impls) == {"query_market_data", "query_news", "query_sec_filings", "query_macro"}
    assert impls["query_news"]({"entity": "TSM", "max_items": 1}, [])["article_count"] == 1
    assert impls["query_macro"]({"indicators": ["vix"]}, [])["indicators"].keys() == {"vix"}


def test_tool_impls_with_extractor_adds_graph_tool(store):
    class FakeExtractor:
        def extract(self, ticker, articles, known):
            return {"events": [{"article_index": 0, "event_type": "SUPPLY_CHAIN", "polarity": "negative",
                                "tickers": [ticker], "summary": f"{ticker} 事件", "confidence": 0.8,
                                "date": "2026-09-12"}], "dropped": {}}

    impls = store.tool_impls(target="AAPL", extractor=FakeExtractor())
    prior_log = [{"tool": "query_news", "result": store.replay_news("AAPL")}]
    result = impls["query_company_graph"]({"entity": "AAPL"}, prior_log)

    assert result["ticker"] == "AAPL"
    assert {n["ticker"] for n in result["neighbors"]} >= {"TSM", "ASML"}
    assert result["propagated_risks"], "邻居的负面事件应产生传导风险"
    assert result["stats"]["extraction"]["companies_ok"]


def test_tool_impls_restrict_applies_to_all_data_tools(store):
    impls = store.tool_impls(target="AAPL", restrict_to="AAPL")
    for tool in ("query_market_data", "query_news", "query_sec_filings"):
        assert "只分析目标公司" in impls[tool]({"entity": "TSM"}, [])["error"]


# ── 其他 ────────────────────────────────────────────────────────

def test_load_companies_skips_comments(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("# 注释\nAAPL\n\n  msft \n", encoding="utf-8")
    assert load_companies(f) == ["AAPL", "MSFT"]


def test_project_companies_file_is_valid():
    targets = load_companies("eval/companies.txt")
    assert len(targets) == 15
    universe, neighbors = universe_for(targets)
    assert all(neighbors[t] for t in targets), "每家目标公司都应有邻居"
    assert len(universe) >= len(targets)
