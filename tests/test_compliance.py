"""
tests/test_compliance.py
-------------------------
compliance.ComplianceChecker：置信度兜底规则 + 合规问题规则 + 免责声明兜底。
纯规则，无网络、无 LLM。
"""

import pytest

from compliance import ComplianceChecker, ensure_disclaimer, DISCLAIMER_MARKER

_GOOD_DRAFT = {
    "executive_summary": "苹果基本面稳健，估值偏高，建议持有。",
    "macro_analysis": "利率维持高位，VIX 处于低位。",
    "fundamental_analysis": "截至 2026-09-15，PE 33 倍，利润率 24%。",
    "technical_analysis": "截至 2026-09-15 股价站上 MA20，RSI 61。",
    "news_sentiment_analysis": "近两周新品发布新闻偏正面。",
    "news_sentiment": "positive",
    "filings_analysis": "7 月底提交 10-Q。",
    "recommendation": "持有",
    "recommendation_rationale": "估值已反映增长预期。",
    "risk_warnings": "估值回调风险、宏观利率风险、监管风险。",
    "confidence": "high",
    "key_signals": ["PE 33 倍（Yahoo Finance）"],
}

_MARKET_OK = {"tool": "query_market_data", "input": {}, "result": {"ticker": "AAPL", "data_as_of": "2026-09-15"}}
_NEWS_OK = {"tool": "query_news", "input": {}, "result": {"articles": []}}
_SEC_OK = {"tool": "query_sec_filings", "input": {}, "result": {"filings": []}}
_MACRO_OK = {"tool": "query_macro", "input": {}, "result": {"summary_text": "..."}}
_ALL_OK = [_MARKET_OK, _NEWS_OK, _SEC_OK, _MACRO_OK]


def _fail(tool):
    return {"tool": tool, "input": {}, "result": {"error": "boom"}}


@pytest.fixture
def checker():
    return ComplianceChecker()


# ── 基线 ────────────────────────────────────────────────────────

def test_clean_report_passes_with_full_score(checker):
    result = checker.check(_GOOD_DRAFT, {"approved": True}, _ALL_OK, "high")

    assert result["is_compliant"] is True
    assert result["score"] == 100
    assert result["issues"] == []
    assert result["draft"]["confidence"] == "high"
    assert result["data_status"] == {
        "query_market_data": "ok", "query_news": "ok", "query_sec_filings": "ok", "query_macro": "ok",
    }


def test_check_does_not_mutate_input_draft(checker):
    draft = dict(_GOOD_DRAFT)
    checker.check(draft, {"confidence_adjustment": "lower"}, _ALL_OK, "high")
    assert draft["confidence"] == "high"


# ── 置信度兜底 ──────────────────────────────────────────────────

def test_critic_lower_is_enforced_when_revision_did_not_lower(checker):
    result = checker.check(_GOOD_DRAFT, {"confidence_adjustment": "lower"}, _ALL_OK, "high")
    assert result["draft"]["confidence"] == "medium"
    assert result["confidence"] == {
        "original": "high", "final": "medium", "reasons": ["Critic 要求下调置信度，修订稿未下调"],
    }
    assert any(i["category"] == "confidence" and i["severity"] == "info" for i in result["issues"])


def test_critic_lower_not_applied_twice_when_revision_already_lowered(checker):
    revised = {**_GOOD_DRAFT, "confidence": "medium"}
    result = checker.check(revised, {"confidence_adjustment": "lower"}, _ALL_OK, "high")
    assert result["draft"]["confidence"] == "medium"


def test_critic_lower_overrides_revision_that_raised_confidence(checker):
    revised = {**_GOOD_DRAFT, "confidence": "high"}
    result = checker.check(revised, {"confidence_adjustment": "lower"}, _ALL_OK, "medium")
    assert result["draft"]["confidence"] == "low"


def test_critic_raise_is_not_applied(checker):
    draft = {**_GOOD_DRAFT, "confidence": "medium"}
    result = checker.check(draft, {"confidence_adjustment": "raise"}, _ALL_OK, "medium")
    assert result["draft"]["confidence"] == "medium"
    assert any("未自动上调" in i["description"] for i in result["issues"])


def test_market_data_missing_caps_confidence_to_low(checker):
    tool_log = [_fail("query_market_data"), _NEWS_OK, _SEC_OK, _MACRO_OK]
    result = checker.check(_GOOD_DRAFT, {}, tool_log, "high")
    assert result["draft"]["confidence"] == "low"
    assert "市场数据" in result["confidence"]["reasons"][0]


def test_one_secondary_source_missing_caps_to_medium(checker):
    tool_log = [_MARKET_OK, _NEWS_OK, _fail("query_sec_filings"), _MACRO_OK]
    result = checker.check(_GOOD_DRAFT, {}, tool_log, "high")
    assert result["draft"]["confidence"] == "medium"


def test_two_secondary_sources_missing_caps_to_low(checker):
    # 宏观根本没调用，也算缺失
    tool_log = [_MARKET_OK, _NEWS_OK, _fail("query_sec_filings")]
    result = checker.check(_GOOD_DRAFT, {}, tool_log, "high")
    assert result["draft"]["confidence"] == "low"
    assert result["data_status"]["query_macro"] == "not_called"


def test_retry_success_counts_as_ok(checker):
    tool_log = [_fail("query_news"), _NEWS_OK, _MARKET_OK, _SEC_OK, _MACRO_OK]
    result = checker.check(_GOOD_DRAFT, {}, tool_log, "high")
    assert result["data_status"]["query_news"] == "ok"
    assert result["draft"]["confidence"] == "high"


def test_invalid_confidence_value_treated_as_low(checker):
    result = checker.check({**_GOOD_DRAFT, "confidence": "very high"}, {}, _ALL_OK, None)
    assert result["draft"]["confidence"] == "low"


# ── 违规表述 ────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase", ["保证收益", "稳赚不赔", "无风险套利", "股价必涨", "据内幕消息", "guaranteed returns", "a risk-free bet"])
def test_prohibited_language_is_critical(checker, phrase):
    draft = {**_GOOD_DRAFT, "recommendation_rationale": f"理由：{phrase}。"}
    result = checker.check(draft, {}, _ALL_OK, "high")

    critical = [i for i in result["issues"] if i["severity"] == "critical"]
    assert len(critical) == 1
    assert critical[0]["field"] == "recommendation_rationale"
    assert result["is_compliant"] is False
    assert result["score"] == 70


@pytest.mark.parametrize("phrase", ["本报告不保证收益", "投资并非无风险", "参考无风险利率 4.2%", "the risk-free rate is 4%"])
def test_negated_or_technical_terms_are_not_flagged(checker, phrase):
    draft = {**_GOOD_DRAFT, "macro_analysis": phrase}
    result = checker.check(draft, {}, _ALL_OK, "high")
    assert result["issues"] == []


def test_prohibited_language_in_key_signals_is_detected(checker):
    draft = {**_GOOD_DRAFT, "key_signals": ["稳赚信号"]}
    result = checker.check(draft, {}, _ALL_OK, "high")
    assert result["issues"][0]["field"] == "key_signals"


# ── 风险提示 / 关键信号 ────────────────────────────────────────

def test_short_risk_warning_and_empty_signals_are_warnings(checker):
    draft = {**_GOOD_DRAFT, "risk_warnings": "有风险", "key_signals": []}
    result = checker.check(draft, {}, _ALL_OK, "high")

    fields = {i["field"] for i in result["issues"] if i["severity"] == "warning"}
    assert fields == {"risk_warnings", "key_signals"}
    assert result["score"] == 80
    assert result["is_compliant"] is True


# ── 数据缺失披露 ────────────────────────────────────────────────

def test_failed_source_must_be_disclosed_in_section(checker):
    tool_log = [_MARKET_OK, _fail("query_news"), _SEC_OK, _MACRO_OK]
    result = checker.check(_GOOD_DRAFT, {}, tool_log, "high")

    warnings = [i for i in result["issues"] if i["severity"] == "warning"]
    assert {i["field"] for i in warnings} == {"news_sentiment_analysis", "news_sentiment"}


def test_disclosed_failure_passes(checker):
    draft = {**_GOOD_DRAFT, "news_sentiment_analysis": "新闻数据不可用（数据源报错）。", "news_sentiment": "unavailable"}
    tool_log = [_MARKET_OK, _fail("query_news"), _SEC_OK, _MACRO_OK]
    result = checker.check(draft, {}, tool_log, "medium")

    assert [i for i in result["issues"] if i["severity"] != "info"] == []


def test_uncalled_source_must_be_disclosed(checker):
    tool_log = [_MARKET_OK, _NEWS_OK, _MACRO_OK]    # 没查 SEC
    result = checker.check({**_GOOD_DRAFT, "confidence": "medium"}, {}, tool_log, "medium")
    issue = next(i for i in result["issues"] if i["field"] == "filings_analysis")
    assert "未查询" in issue["description"]


# ── 日期引用 ────────────────────────────────────────────────────

@pytest.mark.parametrize("text, flagged", [
    ("股价站上 MA20，RSI 61。", True),
    ("截至 2026-09-15 股价站上 MA20。", False),
    ("9 月 15 日股价站上 MA20。", False),
])
def test_analysis_should_cite_data_date(checker, text, flagged):
    draft = {**_GOOD_DRAFT, "technical_analysis": text, "fundamental_analysis": "PE 33 倍。"}
    result = checker.check(draft, {}, _ALL_OK, "high")
    has_issue = any("未注明数据日期" in i["description"] for i in result["issues"])
    assert has_issue is flagged


def test_date_rule_skipped_when_market_data_unavailable(checker):
    draft = {**_GOOD_DRAFT, "technical_analysis": "数据不可用", "fundamental_analysis": "数据不可用"}
    tool_log = [_fail("query_market_data"), _NEWS_OK, _SEC_OK, _MACRO_OK]
    result = checker.check(draft, {}, tool_log, "high")
    assert not any("未注明数据日期" in i["description"] for i in result["issues"])


# ── 建议强度 vs 置信度 ──────────────────────────────────────────

def test_strong_recommendation_with_low_confidence_is_warning(checker):
    tool_log = [_fail("query_market_data")]
    draft = {**_GOOD_DRAFT, "recommendation": "增持",
             "fundamental_analysis": "数据不可用", "technical_analysis": "数据不可用",
             "news_sentiment_analysis": "未能获取", "news_sentiment": "unavailable",
             "filings_analysis": "暂无", "macro_analysis": "无数据"}
    result = checker.check(draft, {}, tool_log, "high")

    assert result["draft"]["confidence"] == "low"       # 先封顶，再判断强度
    assert any(i["category"] == "bias" for i in result["issues"])


# ── 免责声明兜底 ────────────────────────────────────────────────

def test_ensure_disclaimer_appends_when_missing():
    md, added = ensure_disclaimer("# 报告\n正文")
    assert added is True
    assert DISCLAIMER_MARKER in md
    assert md.startswith("# 报告\n正文")


def test_ensure_disclaimer_noop_when_present():
    md, added = ensure_disclaimer(f"# 报告\n## {DISCLAIMER_MARKER}\n...")
    assert added is False
