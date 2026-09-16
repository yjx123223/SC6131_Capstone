"""
tests/test_orchestrator.py
----------------------------
OrchestratorAgent 协调层的行为测试，重点覆盖新增的反馈存储写入：
Multi-Agent 链路生成报告后应记录 session 并返回 session_id，
使其与单 Agent 链路一样支持事后评分。

用假的 loop / critic 替身注入，不发真实 API 请求。
"""

import pytest

from feedback_store import FeedbackStore
from orchestrator import OrchestratorAgent


_DRAFT = {
    "executive_summary": "summary",
    "macro_analysis": "macro",
    "entity_analysis": "entity",
    "recommendation": "增持",
    "recommendation_rationale": "理由",
    "risk_warnings": "风险",
    "confidence": "medium",
    "key_signals": ["signal1"],
}

_KG_TOOL_RESULT = {
    "entity": "Apple Inc.",
    "period": "2022-10-23 ~ 2022-12-25",
    "total_events": 86,
    "positive_impacts": [{"subject": "Goldman Sachs Group", "relation": "Invests_In", "object": "Apple Inc.", "count": 2}],
    "negative_impacts": [{"subject": "Meta Platforms", "relation": "Decrease", "object": "Apple Inc.", "count": 1}],
    "other_relations": [],
    "multihop_context": "（很长的多跳文本，不应写进反馈存储）",
}

_TOOL_LOG = [
    {"tool": "query_kg_signals", "input": {"entity": "Apple Inc."}, "result": _KG_TOOL_RESULT},
    {"tool": "query_macro", "input": {}, "result": {"summary_text": "VIX 20.6"}},
]


class _FakeLoop:
    def __init__(self, draft=_DRAFT, tool_log=None):
        self._draft = draft
        self._tool_log = _TOOL_LOG if tool_log is None else tool_log
        self.model = "fake-model"

    def run(self, entity, weeks, graph, feedback_store=None):
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

    session_id, report_md = orch.generate_report("Apple Inc.", graph=None, feedback_store=store)

    assert isinstance(session_id, int)
    assert "投资建议报告：Apple Inc." in report_md


def test_generate_report_logs_advice_into_store(orch, store, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)

    session_id, report_md = orch.generate_report("Apple Inc.", graph=None, feedback_store=store)

    history = store.get_history("Apple Inc.")
    assert len(history) == 1
    assert history[0]["id"] == session_id
    assert history[0]["total_events"] == 86
    assert history[0]["period"] == "2022-10-23 ~ 2022-12-25"
    assert history[0]["rating"] is None    # 尚未评分


def test_logged_session_feeds_signal_accuracy_report(orch, store, monkeypatch, tmp_path):
    """评分后应能按关系类型聚合——这是反馈闭环回灌给 Agent 的数据来源"""
    _reports_dir(monkeypatch, tmp_path)

    session_id, _ = orch.generate_report("Apple Inc.", graph=None, feedback_store=store)
    store.rate(session_id, rating=1, note="信号准确")

    report = store.signal_accuracy_report()
    assert report["total_rated"] == 1
    assert report["signal_stats"]["Invests_In"]["avg_rating"] == 1.0
    assert report["signal_stats"]["Decrease"]["avg_rating"] == 1.0


def test_generate_report_without_store_returns_none_session(orch, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)

    session_id, report_md = orch.generate_report("Apple Inc.", graph=None, feedback_store=None)

    assert session_id is None
    assert "投资建议报告" in report_md


def test_failed_draft_returns_none_session(orch, store, monkeypatch, tmp_path):
    _reports_dir(monkeypatch, tmp_path)
    orch.loop = _FakeLoop(draft=None)

    session_id, message = orch.generate_report("Apple Inc.", graph=None, feedback_store=store)

    assert session_id is None
    assert "未能生成报告草稿" in message
    assert store.get_history("Apple Inc.") == []


def test_extract_kg_summary_drops_multihop_and_skips_errors():
    tool_log = [
        {"tool": "query_kg_signals", "input": {}, "result": {"error": "未找到数据"}},
        {"tool": "query_kg_signals", "input": {}, "result": _KG_TOOL_RESULT},
    ]
    summary = OrchestratorAgent._extract_kg_summary(tool_log)

    assert summary["total_events"] == 86
    assert "multihop_context" not in summary   # 体积大且不参与统计，应剔除


def test_extract_kg_summary_empty_when_no_kg_tool_called():
    assert OrchestratorAgent._extract_kg_summary([{"tool": "query_macro", "input": {}, "result": {}}]) == {}
