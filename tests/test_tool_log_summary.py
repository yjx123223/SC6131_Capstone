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


def test_macro_and_feedback_and_unknown_tool():
    log = [
        {"tool": "query_macro", "input": {}, "result": {"summary_text": "VIX：15.2"}},
        {"tool": "get_feedback_stats", "input": {}, "result": {"total_rated": 4, "positive_rate": 0.5}},
        {"tool": "some_new_tool", "input": {}, "result": {}},
    ]
    text = summarize_tool_log(log)
    assert "VIX：15.2" in text
    assert "已评4次" in text and "50.0%" in text
    assert "some_new_tool：已调用" in text
