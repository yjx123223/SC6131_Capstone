"""
tests/test_eval_variants.py
----------------------------
eval.variants：三种变体的工具集、emit_report 字段裁剪、prompt 内容与 Agent 组装。
"""

import pytest

import orchestrator_loop as ol
from eval import variants
from eval.snapshot import SnapshotStore, seed_graph
from tests.test_eval_snapshot import _fake_fns


@pytest.fixture
def store():
    return SnapshotStore.record(["AAPL"], on_progress=lambda m: None, **_fake_fns())


@pytest.fixture
def graph():
    return seed_graph()


# ── 工具集 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("variant, has_graph_tool", [("A", False), ("B", False), ("C", True)])
def test_tool_sets(variant, has_graph_tool):
    names = {t["name"] for t in variants.tool_definitions(variant)}
    assert {"query_market_data", "query_news", "query_sec_filings", "query_macro", "emit_report"} <= names
    assert ("query_company_graph" in names) is has_graph_tool


@pytest.mark.parametrize("variant, expected", [
    ("A", {"supply_chain_analysis": False, "cited_event_ids": False}),
    ("B", {"supply_chain_analysis": True, "cited_event_ids": False}),
    ("C", {"supply_chain_analysis": True, "cited_event_ids": True}),
])
def test_emit_report_fields_per_variant(variant, expected):
    emit = next(t for t in variants.tool_definitions(variant) if t["name"] == "emit_report")
    props, required = emit["input_schema"]["properties"], emit["input_schema"]["required"]
    for field, present in expected.items():
        assert (field in props) is present
        assert (field in required) is present
    assert {"executive_summary", "recommendation", "confidence"} <= set(required)


def test_tool_definitions_do_not_mutate_production_constants():
    variants.tool_definitions("A")
    emit = next(t for t in ol.TOOL_DEFINITIONS if t["name"] == "emit_report")
    assert "cited_event_ids" in emit["input_schema"]["properties"]
    assert len(ol.TOOL_DEFINITIONS) == 6


# ── prompt ──────────────────────────────────────────────────────

def test_variant_a_prompt_has_no_supply_chain_hint(graph):
    prompt = variants.system_prompt("A", "AAPL", graph)
    for word in ("供应商", "竞争对手", "query_company_graph", "知识图谱"):
        assert word not in prompt


def test_variant_b_prompt_lists_same_relations_as_graph(graph):
    from live_kg.queries import find_neighbors

    prompt = variants.system_prompt("B", "AAPL", graph)
    assert "query_company_graph" not in prompt
    for n in find_neighbors(graph, "AAPL"):
        assert n["ticker"] in prompt          # B 组拿到与 C 组相同的关系信息
    assert "经由 TSM" in prompt                # 2 跳也给出
    assert "供应商：" in prompt and "竞争对手：" in prompt
    assert "属于推断" in prompt


def test_variant_c_prompt_is_production_prompt(graph):
    assert variants.system_prompt("C", "AAPL", graph) is ol.SYSTEM_PROMPT


def test_relations_text_for_company_without_neighbors(graph):
    assert "没有相关公司" in variants.relations_text("IBM", graph)


# ── 组装 Agent ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fake_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: object())


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_build_agent_wires_variant(store, graph, variant):
    agent = variants.build_agent(variant, store, "AAPL", graph=graph)

    assert agent.loop.temperature == 0 and agent.critic.temperature == 0
    assert {t["name"] for t in agent.loop.tool_definitions} == \
        {t["name"] for t in variants.tool_definitions(variant)}
    assert agent.loop.system_prompt == variants.system_prompt(variant, "AAPL", graph)
    assert ("query_company_graph" in agent.loop.tool_impls) is (variant == "C")
    # 所有数据工具都走快照，不会打到真实接口
    assert agent.loop.tool_impls["query_news"]({"entity": "AAPL"}, [])["ticker"] == "AAPL"


def test_variant_a_agent_cannot_query_neighbors(store, graph):
    agent = variants.build_agent("A", store, "AAPL", graph=graph)
    assert "只分析目标公司" in agent.loop.tool_impls["query_news"]({"entity": "TSM"}, [])["error"]


def test_variant_b_agent_can_query_neighbors(store, graph):
    agent = variants.build_agent("B", store, "AAPL", graph=graph)
    assert agent.loop.tool_impls["query_news"]({"entity": "TSM"}, [])["ticker"] == "TSM"


def test_unknown_variant_raises(store, graph):
    with pytest.raises(ValueError):
        variants.build_agent("D", store, "AAPL", graph=graph)
