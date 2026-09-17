"""
tests/test_tool_log_summary.py
-------------------------------
tool_log_summary.summarize_tool_log：Critic / revise 看到的原始数据摘要。
"""

from tool_log_summary import summarize_tool_log


def test_empty_log():
    assert summarize_tool_log([]) == "（无工具调用记录）"


def test_market_summary_includes_date_and_key_numbers():
    log = [{"tool": "query_market_data", "input": {}, "result": {
        "ticker": "AAPL", "data_as_of": "2026-09-15",
        "technicals": {"last_close": 229.0, "rsi14": 72.1, "rsi14_signal": "overbought"},
        "fundamentals": {"trailing_pe": 33.5},
        "warnings": ["公司信息/财务指标为空"],
    }}]
    text = summarize_tool_log(log)
    assert "AAPL" in text and "2026-09-15" in text
    assert "229.0" in text and "72.1" in text and "overbought" in text and "33.5" in text
    assert "公司信息/财务指标为空" in text


def test_news_summary_lists_titles_with_dates():
    log = [{"tool": "query_news", "input": {}, "result": {
        "ticker": "AAPL", "lookback_days": 14,
        "articles": [{"title": "Apple launches iPhone", "publisher": "Reuters",
                      "published_at": "2026-09-10T10:00+00:00"}],
    }}]
    text = summarize_tool_log(log)
    assert "1 条" in text
    assert "2026-09-10 Reuters: Apple launches iPhone" in text


def test_sec_summary_lists_forms():
    log = [{"tool": "query_sec_filings", "input": {}, "result": {
        "ticker": "AAPL", "filings": [{"form": "10-Q", "filing_date": "2026-07-31"}],
    }}]
    assert "10-Q(2026-07-31)" in summarize_tool_log(log)


def test_errors_are_marked_unavailable():
    log = [{"tool": "query_sec_filings", "input": {}, "result": {"error": "未配置 SEC_EDGAR_USER_AGENT"}}]
    text = summarize_tool_log(log)
    assert "数据不可用" in text and "SEC_EDGAR_USER_AGENT" in text


def test_macro_and_unknown_tool():
    log = [
        {"tool": "query_macro", "input": {}, "result": {"summary_text": "VIX：15.2"}},
        {"tool": "some_new_tool", "input": {}, "result": {}},
    ]
    text = summarize_tool_log(log)
    assert "VIX：15.2" in text
    assert "some_new_tool：已调用" in text


def test_graph_summary_lists_neighbors_events_and_risks():
    log = [{"tool": "query_company_graph", "input": {}, "result": {
        "ticker": "AAPL",
        "stats": {"node_count": 9, "edge_count": 11},
        "neighbors": [{"ticker": "TSM", "roles": ["供应商"], "via": None},
                      {"ticker": "ASML", "roles": ["上游供应商"], "via": "TSM"}],
        "target_events": [{"id": "E1", "date": "2026-09-10", "polarity": "positive",
                           "event_type": "PRODUCT", "summary": "iPhone 18 发布"}],
        "propagated_risks": [{"event_id": "E2", "impact": "供应风险", "score": 0.8, "date": "2026-09-12",
                              "neighbor": "TSM", "summary": "产能受限", "path": "TSM ─SUPPLIES_TO→ AAPL"}],
        "opportunities": [{"event_id": "E3", "impact": "竞争对手承压", "neighbor": "MSFT", "summary": "承压"}],
        "warnings": ["抽取失败 ASML"],
        "_graph": {"nodes": ["不应出现"]},
    }}]
    text = summarize_tool_log(log)
    assert "知识图谱[AAPL]：9 个节点、11 条边；邻居 TSM(供应商), ASML(上游供应商, 经 TSM)" in text
    assert "目标事件 E1 2026-09-10 positive PRODUCT：iPhone 18 发布" in text
    assert "传导风险 E2 供应风险（分数 0.8）" in text and "路径 TSM ─SUPPLIES_TO→ AAPL" in text
    assert "潜在利好 E3" in text
    assert "图谱警告：抽取失败 ASML" in text
    assert "不应出现" not in text


def test_latest_result_returns_last_success():
    from tool_log_summary import latest_result
    log = [
        {"tool": "query_news", "result": {"n": 1}},
        {"tool": "query_news", "result": {"n": 2}},
        {"tool": "query_news", "result": {"error": "x"}},
    ]
    assert latest_result(log, "query_news") == {"n": 2}
    assert latest_result(log, "query_macro") is None
