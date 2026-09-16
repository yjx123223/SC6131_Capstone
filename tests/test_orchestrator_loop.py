"""
tests/test_orchestrator_loop.py
----------------------------------
OrchestratorLoop 的行为测试（拆分自 orchestrator.py 的
OrchestratorAgent._run_orchestrator_loop / _execute_tool / revise）。

多轮 tool-use 循环什么时候终止、工具调用怎么分发（含同轮并行调用、
失败工具的 is_error 标记、已停用的图谱工具）、revise 修订。
用一个按顺序回放响应的 stub client 驱动，实时数据工具用 monkeypatch 替换，
不发真实网络请求。
"""

import pytest

import orchestrator_loop
from orchestrator_loop import OrchestratorLoop, TOOL_DEFINITIONS, SYSTEM_PROMPT


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
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses[self.call_count]
        self.call_count += 1
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses):
        self.messages = _ScriptedMessages(responses)


_MARKET_RESULT = {"ticker": "AAPL", "data_as_of": "2026-09-15", "technicals": {}, "fundamentals": {}}


@pytest.fixture
def fake_tools(monkeypatch):
    """把三个实时数据工具替换成记录参数的假实现，不联网"""
    calls = {}

    def market(entity, period="3mo"):
        calls["market"] = {"entity": entity, "period": period}
        return dict(_MARKET_RESULT)

    def news(entity, max_items=8):
        calls["news"] = {"entity": entity, "max_items": max_items}
        return {"error": "新闻源不可用"}

    def sec(entity, form_types=None):
        calls["sec"] = {"entity": entity, "form_types": form_types}
        return {"ticker": "AAPL", "filings": []}

    monkeypatch.setattr(orchestrator_loop.market_tools, "query_market_data", market)
    monkeypatch.setattr(orchestrator_loop.news_tools, "query_news", news)
    monkeypatch.setattr(orchestrator_loop.sec_tools, "query_sec_filings", sec)
    return calls


_EMIT_INPUT = {
    "executive_summary": "summary",
    "macro_analysis": "macro",
    "fundamental_analysis": "fundamentals",
    "technical_analysis": "technicals",
    "news_sentiment_analysis": "news",
    "news_sentiment": "neutral",
    "filings_analysis": "filings",
    "recommendation": "增持",
    "recommendation_rationale": "理由",
    "risk_warnings": "风险",
    "confidence": "medium",
    "key_signals": ["signal1"],
}


def test_run_calls_market_tool_then_emits_report(fake_tools):
    responses = [
        _FakeResponse(
            content=[_FakeToolUseBlock("query_market_data", {"entity": "Apple Inc."}, "tool_1")],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[_FakeToolUseBlock("emit_report", _EMIT_INPUT, "tool_2")],
            stop_reason="tool_use",
        ),
    ]
    client = _FakeClient(responses)
    loop = OrchestratorLoop(client, model="m", max_tokens=100)

    draft, tool_log = loop.run("Apple Inc.")

    assert draft == _EMIT_INPUT
    assert len(tool_log) == 1
    assert tool_log[0]["tool"] == "query_market_data"
    assert tool_log[0]["result"]["ticker"] == "AAPL"
    assert fake_tools["market"] == {"entity": "Apple Inc.", "period": "3mo"}   # 未传 period 用默认值
    assert client.messages.call_count == 2


def test_parallel_tool_calls_in_one_turn_and_error_flag(fake_tools):
    """同一轮多个 tool_use：每个都要有对应 tool_result；失败的工具标记 is_error"""
    responses = [
        _FakeResponse(
            content=[
                _FakeToolUseBlock("query_market_data", {"entity": "AAPL", "period": "1y"}, "t1"),
                _FakeToolUseBlock("query_news", {"entity": "AAPL", "max_items": 5}, "t2"),
                _FakeToolUseBlock("query_sec_filings", {"entity": "AAPL", "form_types": ["8-K"]}, "t3"),
            ],
            stop_reason="tool_use",
        ),
        _FakeResponse(content=[_FakeToolUseBlock("emit_report", _EMIT_INPUT, "t4")], stop_reason="tool_use"),
    ]
    client = _FakeClient(responses)
    loop = OrchestratorLoop(client, model="m", max_tokens=100)

    draft, tool_log = loop.run("AAPL")

    assert [e["tool"] for e in tool_log] == ["query_market_data", "query_news", "query_sec_filings"]
    assert fake_tools["market"]["period"] == "1y"
    assert fake_tools["news"]["max_items"] == 5
    assert fake_tools["sec"]["form_types"] == ["8-K"]

    messages = client.messages.calls[-1]["messages"]
    tool_result_msg = messages[2]
    assert tool_result_msg["role"] == "user"
    blocks = {b["tool_use_id"]: b for b in tool_result_msg["content"]}
    assert set(blocks) == {"t1", "t2", "t3"}
    assert blocks["t2"].get("is_error") is True
    assert "is_error" not in blocks["t1"]


def test_kg_tool_is_disabled_and_not_dispatched():
    """图谱工具已停用：schema 中不再暴露，模型若仍调用则得到 error 而不是真的查询图谱"""
    loop = OrchestratorLoop(_FakeClient([]), model="m", max_tokens=100)
    result = loop._execute_tool("query_kg_signals", {"entity": "Apple Inc."}, context={"graph": object()})
    assert "未知工具" in result["error"]


def test_tool_exception_is_converted_to_error(monkeypatch):
    def boom(entity, period="3mo"):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(orchestrator_loop.market_tools, "query_market_data", boom)
    loop = OrchestratorLoop(_FakeClient([]), model="m", max_tokens=100)

    result = loop._execute_tool("query_market_data", {"entity": "AAPL"}, context={})
    assert "unexpected" in result["error"]


def test_initial_message_contains_today_and_entity(fake_tools, monkeypatch):
    monkeypatch.setattr(OrchestratorLoop, "_today", staticmethod(lambda: "2026-09-16"))
    client = _FakeClient([_FakeResponse(content=[], stop_reason="end_turn")])
    OrchestratorLoop(client, model="m", max_tokens=100).run("Tesla Inc.")

    call = client.messages.calls[0]
    first_user = call["messages"][0]["content"]
    assert "2026-09-16" in first_user
    assert "Tesla Inc." in first_user
    assert call["system"] == SYSTEM_PROMPT


def test_run_stops_on_end_turn_without_emit_report():
    responses = [
        _FakeResponse(content=[], stop_reason="end_turn"),
    ]
    client = _FakeClient(responses)
    loop = OrchestratorLoop(client, model="m", max_tokens=100)

    draft, tool_log = loop.run("Apple Inc.")

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

    draft, tool_log = loop.run("Apple Inc.")

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


def test_tool_definitions_expose_realtime_tools_without_kg():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert names == {
        "query_market_data", "query_news", "query_sec_filings",
        "query_macro", "get_feedback_stats", "emit_report",
    }


def test_emit_report_schema_requires_all_draft_fields():
    from report_fields import DRAFT_FIELDS

    emit = next(t for t in TOOL_DEFINITIONS if t["name"] == "emit_report")
    required = set(emit["input_schema"]["required"])
    assert {k for k, _ in DRAFT_FIELDS} <= required
    assert {"news_sentiment", "key_signals"} <= required
    assert "entity_analysis" not in emit["input_schema"]["properties"]


class _FakeTextOnlyResponse:
    def __init__(self):
        self.content = [type("B", (), {"type": "text", "text": "no tool"})()]
        self.stop_reason = "end_turn"


def test_revise_forces_emit_report_and_returns_new_draft():
    revised = {**_EMIT_INPUT, "confidence": "low"}
    client = _FakeClient([_FakeResponse(content=[_FakeToolUseBlock("emit_report", revised, "r1")], stop_reason="tool_use")])
    loop = OrchestratorLoop(client, model="m", max_tokens=100)
    critique = {"conflicts": ["置信度过高"], "suggestions": "降为 low"}
    tool_log = [{"tool": "query_market_data", "input": {}, "result": {"error": "403"}}]

    result = loop.revise("Apple Inc.", _EMIT_INPUT, critique, tool_log)

    assert result == revised
    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "emit_report"}
    prompt = call["messages"][0]["content"]
    assert "置信度过高" in prompt
    assert "数据不可用" in prompt          # 原始数据摘要（含失败的工具）进入修订 prompt


def test_revise_falls_back_to_original_draft():
    loop = OrchestratorLoop(_FakeClient([_FakeTextOnlyResponse()]), model="m", max_tokens=100)
    assert loop.revise("Apple Inc.", _EMIT_INPUT, {}, []) == _EMIT_INPUT

    loop = OrchestratorLoop(_FakeClient([RuntimeError("API down")]), model="m", max_tokens=100)
    assert loop.revise("Apple Inc.", _EMIT_INPUT, {}, []) == _EMIT_INPUT
