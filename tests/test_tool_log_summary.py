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

    log[0]["result"]["filings"][0]["report_date"] = "2026-06-27"
    assert "10-Q(2026-07-31，报告期 2026-06-27)" in summarize_tool_log(log)


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


# ── 回归：摘要漏字段会导致 Critic 误判"编造"、revise 删掉真实内容 ──────

def test_market_summary_keeps_every_fundamental_field():
    """摘要漏掉的基本面字段，Critic 会当成编造要求删除（实测 MSFT 的 ROE/净利增速）"""
    fundamentals = {
        "market_cap": 3666585845760, "trailing_pe": 27.69, "forward_pe": 20.93,
        "price_to_book": 8.29, "dividend_yield_pct": 0.79, "profit_margin_pct": 40.3,
        "revenue_growth_pct": 17.7, "earnings_growth_pct": 31.7,
        "return_on_equity_pct": 34.04, "debt_to_equity": 29.12, "current_ratio": 1.23,
        "total_revenue": 331839012864, "net_income": 133748998144,
        "free_cash_flow": 16545500160, "beta": 1.11, "fifty_two_week_high": 553.72,
        "fifty_two_week_low": 349.2, "analyst_target_mean": 572.92,
        "analyst_recommendation": "strong_buy", "analyst_count": 52,
    }
    log = [{"tool": "query_market_data", "input": {}, "result": {
        "ticker": "MSFT", "data_as_of": "2026-09-18",
        "technicals": {"last_close": 493.78}, "fundamentals": fundamentals,
    }}]
    text = summarize_tool_log(log)
    for value in fundamentals.values():
        assert str(value) in text, f"基本面字段 {value} 未进入摘要"


def test_market_summary_omits_missing_fields():
    log = [{"tool": "query_market_data", "input": {}, "result": {
        "ticker": "AAPL", "technicals": {}, "fundamentals": {"trailing_pe": 30.0, "beta": None},
    }}]
    text = summarize_tool_log(log)
    assert "PE(TTM) 30.0" in text and "beta" not in text


def test_news_summary_keeps_all_articles_with_body():
    """曾只取前 6 条标题、丢掉正文摘要：Evercore $380 就藏在正文里"""
    arts = [{"title": f"标题{i}", "publisher": "P", "published_at": f"2026-09-{10 + i}T00:00",
             "summary": f"正文{i}", "url": ""} for i in range(8)]
    arts[7]["summary"] = "Evercore Delivers Bullish $380 Price Target for Apple Stock"
    log = [{"tool": "query_news", "input": {}, "result": {
        "ticker": "AAPL", "lookback_days": 14, "articles": arts}}]

    text = summarize_tool_log(log)
    assert "近14天 8 条" in text
    for i in range(8):
        assert f"标题{i}" in text                     # 不截断条数
    assert "$380" in text                             # 正文摘要必须在
    assert "Evercore" in text


def test_news_summary_handles_article_without_summary():
    log = [{"tool": "query_news", "input": {}, "result": {
        "ticker": "AAPL", "lookback_days": 14,
        "articles": [{"title": "只有标题", "publisher": "P", "published_at": "2026-09-18T00:00"}]}}]
    text = summarize_tool_log(log)
    assert "只有标题" in text and "摘要：" not in text
