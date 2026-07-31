"""
tests/test_kg_query.py
------------------------
FinDKGGraph 基础查询行为的基线测试，使用 conftest.py 里的
tiny_findkg_dir 构造的最小数据集，不依赖外部真实 FinDKG 数据。
"""

import pytest

from kg_query import FinDKGGraph


def test_stats(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    stats = g.stats()
    assert stats["entities"] == 3
    assert stats["relations"] == 3
    assert stats["triples"] == 4


def test_fuzzy_search(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    assert "Apple Inc." in g.fuzzy_search("apple")
    assert g.fuzzy_search("nonexistent") == []


def test_query_entity_unknown_raises(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    with pytest.raises(ValueError):
        g.query_entity("Nonexistent Corp")


def test_query_entity_returns_all_involving_rows(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    df = g.query_entity("Apple Inc.", n_recent_weeks=3)
    # Apple 出现在全部 4 条三元组中（3 次作宾语 + 1 次作主语）
    assert len(df) == 4


def test_query_entity_respects_recent_weeks_window(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    # 只看最近 1 周（max_time_id=2 时只剩 t=2 那一条）
    df = g.query_entity("Apple Inc.", n_recent_weeks=1)
    assert len(df) == 1
    assert df.iloc[0]["date"] == "2022-01-17"


def test_get_impact_summary_categorizes_signals(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    summary = g.get_impact_summary("Apple Inc.", n_recent_weeks=3)

    assert summary["total_events"] == 4
    # Goldman->Apple Positive_Impact_On, Apple->Goldman Positive_Impact_On,
    # Goldman->Apple Raise 都属于正面（Raise 也在 positive_rels 集合里）
    positive_relations = {p["relation"] for p in summary["positive_impacts"]}
    assert "Positive_Impact_On" in positive_relations
    assert "Raise" in positive_relations

    negative_relations = {n["relation"] for n in summary["negative_impacts"]}
    assert "Negative_Impact_On" in negative_relations


def test_get_impact_summary_zero_events_when_window_too_narrow(tiny_findkg_dir):
    g = FinDKGGraph(data_dir=tiny_findkg_dir)
    # n_recent_weeks=0 会把窗口推到 max_time_id 之外，此时该实体没有事件，
    # 这是 kg_tools.query_kg_signals 判断 "KG 中未找到近期数据" 的依据。
    summary = g.get_impact_summary("Apple Inc.", n_recent_weeks=0)
    assert summary["total_events"] == 0
