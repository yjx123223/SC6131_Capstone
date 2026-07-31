"""
tests/test_kg_tools.py
------------------------
tools.kg_tools.query_kg_signals 的行为测试。

这是被 orchestrator_loop.py 和 mcp_servers/kg_server.py 共用的唯一实现，
用 stub graph（不依赖真实数据集）验证：
  - 正常返回结构
  - 无数据时返回 error
  - 事件截断到 6 条
  - 显式传入 graph 时不会触发 tools.resources 的单例加载

注：predictor / predictions / has_predictions 已从 Multi-Agent 主链路
摘除（kg_predictor 只保留给单 Agent 链路用），相关测试一并移除。
"""

import pytest

from tools import kg_tools, resources


class FakeGraph:
    """最小可用的 graph 替身：只实现 get_impact_summary"""

    def __init__(self, summary: dict):
        self._summary = summary

    def get_impact_summary(self, entity: str, n_recent_weeks: int = 12) -> dict:
        return self._summary


_SAMPLE_SUMMARY = {
    "entity": "Apple Inc.",
    "period": "2022-01-03 ~ 2022-01-17",
    "total_events": 4,
    "positive_impacts": [
        {"subject": "Goldman Sachs Group", "relation": "Positive_Impact_On", "object": "Apple Inc.", "count": 1},
    ],
    "negative_impacts": [
        {"subject": "Meta Platforms", "relation": "Negative_Impact_On", "object": "Apple Inc.", "count": 1},
    ],
    "other_relations": [],
}


def test_query_kg_signals_happy_path():
    graph = FakeGraph(_SAMPLE_SUMMARY)

    result = kg_tools.query_kg_signals("Apple Inc.", weeks=12, graph=graph)

    assert result["entity"] == "Apple Inc."
    assert result["total_events"] == 4
    assert result["positive_impacts"][0]["relation"] == "Positive_Impact_On"
    assert result["negative_impacts"][0]["relation"] == "Negative_Impact_On"
    # predictor 已从主链路摘除，返回结构中不应再有预测相关字段
    assert "predictions" not in result
    assert "has_predictions" not in result


def test_query_kg_signals_no_events_returns_error():
    graph = FakeGraph({"entity": "Unknown Corp", "total_events": 0, "period": "最近12周"})
    result = kg_tools.query_kg_signals("Unknown Corp", weeks=12, graph=graph)

    assert "error" in result


def test_query_kg_signals_truncates_events_to_six():
    many_events = [
        {"subject": "S", "relation": "Positive_Impact_On", "object": "Apple Inc.", "count": i}
        for i in range(10)
    ]
    summary = {**_SAMPLE_SUMMARY, "positive_impacts": many_events, "total_events": 10}
    graph = FakeGraph(summary)

    result = kg_tools.query_kg_signals("Apple Inc.", graph=graph)

    assert len(result["positive_impacts"]) == 6


def test_explicit_graph_bypasses_resources_singleton(monkeypatch):
    """
    显式传入 graph 时，不应该触发 resources.get_graph()
    （避免重复加载整个数据集）。
    """
    def _boom():
        raise AssertionError("不应该调用 resources 的单例加载")

    monkeypatch.setattr(resources, "get_graph", _boom)

    graph = FakeGraph(_SAMPLE_SUMMARY)

    # 不应抛出 AssertionError
    result = kg_tools.query_kg_signals("Apple Inc.", graph=graph)
    assert "error" not in result
