"""
tests/test_orchestrator.py
----------------------------
OrchestratorAgent 协调层的行为测试：
  - 生成报告后记录 session 并返回 session_id（支持事后评分）
  - 反馈存储里保存的是实时数据快照（图谱工具已停用）
  - Critic 不通过时触发修订

用假的 loop / critic 替身注入，不发真实 API 请求。
"""

import pytest

from feedback_store import FeedbackStore
from orchestrator import OrchestratorAgent


_DRAFT = {
    "executive_summary": "summary",
    "macro_analysis": "macro",
    "fundamental_analysis": "fundamentals",
    "technical_analysis": "technicals",
    "news_sentiment_analysis": "news",
    "news_sentiment": "positive",
    "filings_analysis": "filings",
    "recommendation": "增持",
    "recommendation_rationale": "理由",
    "risk_warnings": "风险",
    "confidence": "medium",
    "key_signals": ["signal1"],
}

_MARKET_RESULT = {
    "ticker": "AAPL",
    "data_as_of": "2026-09-15",
    "technicals": {"last_close": 229.0, "rsi14": 61.2},
    "fundamentals": {"trailing_pe": 33.5, "forward_pe": 29.0, "profit_margin_pct": 24.3,
                     "revenue_growth_pct": 6.1, "analyst_target_mean": 250.0, "beta": 1.2},
    "recent_closes": [{"date": "2026-09-15", "close": 229.0}],
    "warnings": [],
}

_NEWS_RESULT = {
    "ticker": "AAPL",
    "article_count": 3,
    "articles": [{"title": "t", "publisher": "p", "published_at": "2026-09-15T10:00", "summary": "", "url": ""}] * 3,
}

_SEC_RESULT = {
    "ticker": "AAPL",
    "filings": [{"form": "10-Q", "filing_date": "2026-07-31", "url": "u"}],
}

_TOOL_LOG = [
    {"tool": "query_market_data", "input": {"entity": "Apple Inc."}, "result": _MARKET_RESULT},
    {"tool": "query_news", "input": {"entity": "Apple Inc."}, "result": _NEWS_RESULT},
    {"tool": "query_sec_filings", "input": {"entity": "Apple Inc."}, "result": _SEC_RESULT},
    {"tool": "query_macro", "input": {}, "result": {"error": "未配置 FRED_API_KEY"}},
]


class _FakeLoop:
    def __init__(self, draft=_DRAFT, tool_log=None):
        self._draft = draft
        self._tool_log = _TOOL_LOG if tool_log is None else tool_log
        self.model = "fake-model"

    def run(self, entity, weeks=12, graph=None, feedback_store=None):
        return self._draft, self._tool_log

    def revise(self, entity, draft, critique, tool_log):
        return draft


class _FakeCritic:
    def review(self, entity, draft, tool_log):
        return {"approved": True, "conflicts": [], "confidence_adjustment": "maintain", "suggestions": ""}


@pytest.fixture
def orch(monkeypatch, tmp_path):
    """构造一个不含真实 anthropic client 的 OrchestratorAgent"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: object())

    agent = OrchestratorAgent()
    agent.loop = _FakeLoop()
    agent.critic = _FakeCritic()
    return agent


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "feedback.db")


def _reports_dir(monkeypatch, tmp_path):
    """把报告输出目录指到临时目录，避免污染项目的 reports/"""
    import config
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")


def test_generate_report_returns_session_id_and_markdown(orch, store, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)

    session_id, report_md = orch.generate_report("Apple Inc.", feedback_store=store)

    assert isinstance(session_id, int)
    assert "投资建议报告：Apple Inc." in report_md


def test_generate_report_logs_advice_into_store(orch, store, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)

    session_id, report_md = orch.generate_report("Apple Inc.", feedback_store=store)

    history = store.get_history("Apple Inc.")
    assert len(history) == 1
    assert history[0]["id"] == session_id
    assert history[0]["period"] == "行情截至 2026-09-15"   # period 列存行情截至日期
    assert history[0]["total_events"] == 3                 # total_events 列存新闻条数
    assert history[0]["kg_summary"]["ticker"] == "AAPL"
    assert history[0]["rating"] is None    # 尚未评分


def test_rating_new_style_session_does_not_break_accuracy_report(orch, store, monkeypatch, tmp_path):
    """新快照里没有 KG 关系字段，signal_accuracy_report 应正常返回而不是报错"""
    _reports_dir(monkeypatch, tmp_path)

    session_id, _ = orch.generate_report("Apple Inc.", feedback_store=store)
    store.rate(session_id, rating=1, note="判断准确")

    report = store.signal_accuracy_report()
    assert report["total_rated"] == 1
    assert report["positive_rate"] == 1.0
    assert report["signal_stats"] == {}


def test_generate_report_without_store_returns_none_session(orch, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)

    session_id, report_md = orch.generate_report("Apple Inc.", feedback_store=None)

    assert session_id is None
    assert "投资建议报告" in report_md


def test_generate_report_graph_is_optional(orch, monkeypatch, tmp_path):
    """图谱工具停用后，generate_report 不再要求传入 graph"""
    _reports_dir(monkeypatch, tmp_path)
    session_id, report_md = orch.generate_report("Apple Inc.")
    assert "投资建议报告：Apple Inc." in report_md


def test_failed_draft_returns_none_session(orch, store, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)
    orch.loop = _FakeLoop(draft=None)

    session_id, message = orch.generate_report("Apple Inc.", feedback_store=store)

    assert session_id is None
    assert "未能生成报告草稿" in message
    assert store.get_history("Apple Inc.") == []


def test_revise_is_called_when_critic_rejects(orch, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)
    revised = {**_DRAFT, "confidence": "low", "executive_summary": "修订后的摘要"}
    calls = []

    class RejectingCritic:
        def review(self, entity, draft, tool_log):
            return {"approved": False, "conflicts": ["置信度过高"],
                    "confidence_adjustment": "lower", "suggestions": "降低置信度"}

    def fake_revise(entity, draft, critique, tool_log):
        calls.append(critique)
        return revised

    orch.critic = RejectingCritic()
    monkeypatch.setattr(orch.loop, "revise", fake_revise)

    _, report_md = orch.generate_report("Apple Inc.")

    assert len(calls) == 1
    assert "修订后的摘要" in report_md
    assert "置信度过高" in report_md


def test_generate_report_applies_compliance_confidence_cap(orch, store, monkeypatch, tmp_path):
    """宏观数据失败（_TOOL_LOG 中 query_macro 报错）→ 置信度最高 medium；原稿 high 应被下调"""
    _reports_dir(monkeypatch, tmp_path)
    orch.loop = _FakeLoop(draft={**_DRAFT, "confidence": "high"})

    session_id, report_md = orch.generate_report("Apple Inc.", feedback_store=store)

    assert "## 合规检查" in report_md
    assert "原为高，已按审查/合规规则下调" in report_md
    assert "## 免责声明" in report_md
    snap = store.get_history("Apple Inc.")[0]["kg_summary"]
    assert snap["confidence"] == "medium"
    assert snap["confidence_original"] == "high"
    assert isinstance(snap["compliance_score"], int)
    assert "is_compliant" in snap


def test_saved_report_matches_returned_markdown(orch, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)
    _, report_md = orch.generate_report("Apple Inc.")
    saved = list((tmp_path / "reports").glob("Apple_Inc*.md"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == report_md


def test_extract_signal_snapshot_collects_key_fields():
    snap = OrchestratorAgent._extract_signal_snapshot(_TOOL_LOG, _DRAFT)

    assert snap["ticker"] == "AAPL"
    assert snap["period"] == "行情截至 2026-09-15"
    assert snap["total_events"] == 3
    assert snap["recommendation"] == "增持"
    assert snap["news_sentiment"] == "positive"
    assert snap["sec_forms"] == ["10-Q 2026-07-31"]
    assert snap["tools_ok"] == ["query_market_data", "query_news", "query_sec_filings"]
    assert snap["tools_failed"] == ["query_macro"]
    # 只存关键财务数值，不存全部字段
    assert "beta" not in snap["fundamentals"]
    # 新闻正文等大字段不应进入快照
    assert "articles" not in snap


def test_extract_signal_snapshot_when_all_tools_failed():
    tool_log = [{"tool": "query_market_data", "input": {}, "result": {"error": "403"}}]
    snap = OrchestratorAgent._extract_signal_snapshot(tool_log, _DRAFT)

    assert snap["period"] == ""
    assert snap["total_events"] == 0
    assert snap["tools_failed"] == ["query_market_data"]
    assert "ticker" not in snap
