"""
tests/test_orchestrator_loop.py
----------------------------------
OrchestratorLoop 的行为测试（拆分自 orchestrator.py 的
OrchestratorAgent._run_orchestrator_loop / _execute_tool / revise）。

这是重构前完全没有测试覆盖、风险最高的一段：多轮 tool-use 循环
什么时候终止、工具调用怎么分发。用一个按顺序回放响应的 stub
client 驱动，不发真实网络请求。
"""

import pytest

from orchestrator_loop import OrchestratorLoop, TOOL_DEFINITIONS


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name, input_, id_):
        self.name = name
        self.input = input_
        self.id = id_


class _FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _ScriptedMessages:
    """按顺序回放一串预先准备好的响应，每次 create() 吐出下一个"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


class _FakeClient:
    def __init__(self, responses):
        self.messages = _ScriptedMessages(responses)


class FakeGraph:
    def get_impact_summary(self, entity, n_recent_weeks=12):
        return {
            "entity": entity,
            "period": "2022-01-03 ~ 2022-01-17",
            "total_events": 3,
            "positive_impacts": [{"subject": "Goldman Sachs Group", "relation": "Positive_Impact_On", "object": entity, "count": 1}],
            "negative_impacts": [],
            "other_relations": [],
        }


_EMIT_INPUT = {
    "executive_summary": "summary",
    "macro_analysis": "macro",
    "entity_analysis": "entity",
    "recommendation": "增持",
    "recommendation_rationale": "理由",
    "risk_warnings": "风险",
    "confidence": "medium",
    "key_signals": ["signal1"],
}


def test_run_calls_kg_tool_then_emits_report():
    responses = [
        _FakeResponse(
            content=[_FakeToolUseBlock("query_kg_signals", {"entity": "Apple Inc.", "weeks": 12}, "tool_1")],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[_FakeToolUseBlock("emit_report", _EMIT_INPUT, "tool_2")],
            stop_reason="tool_use",
        ),
    ]
    client = _FakeClient(responses)
    loop = OrchestratorLoop(client, model="m", max_tokens=100)

    draft, tool_log = loop.run("Apple Inc.", weeks=12, graph=FakeGraph())

    assert draft == _EMIT_INPUT
    assert len(tool_log) == 1
    assert tool_log[0]["tool"] == "query_kg_signals"
    assert tool_log[0]["result"]["entity"] == "Apple Inc."
    assert client.messages.call_count == 2


def test_run_stops_on_end_turn_without_emit_report():
    responses = [
        _FakeResponse(content=[], stop_reason="end_turn"),
    ]
    client = _FakeClient(responses)
    loop = OrchestratorLoop(client, model="m", max_tokens=100)

    draft, tool_log = loop.run("Apple Inc.", weeks=12, graph=FakeGraph())

    assert draft is None
    assert tool_log == []


def test_run_respects_max_iterations_when_model_never_emits():
    # 每一轮都调用一个无关工具，永远不调 emit_report
    responses = [
        _FakeResponse(
            content=[_FakeToolUseBlock("get_feedback_stats", {}, f"tool_{i}")],
            stop_reason="tool_use",
        )
        for i in range(OrchestratorLoop.MAX_ITERATIONS)
    ]
    client = _FakeClient(responses)
    loop = OrchestratorLoop(client, model="m", max_tokens=100)

    draft, tool_log = loop.run("Apple Inc.", weeks=12, graph=FakeGraph())

    assert draft is None
    assert client.messages.call_count == OrchestratorLoop.MAX_ITERATIONS


def test_query_macro_tool_uses_configured_fred_key():
    loop = OrchestratorLoop(_FakeClient([]), model="m", max_tokens=100, fred_api_key=None)
    result = loop._tool_query_macro({})
    assert "error" in result  # 没配置 FRED_API_KEY，应该报错而不是抛异常


def test_feedback_stats_tool_without_store_returns_error():
    loop = OrchestratorLoop(_FakeClient([]), model="m", max_tokens=100)
    result = loop._tool_feedback_stats({}, context={})
    assert "error" in result


def test_tool_definitions_include_all_four_tools():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert names == {"query_kg_signals", "query_macro", "get_feedback_stats", "emit_report"}
