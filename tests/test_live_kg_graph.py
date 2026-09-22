"""
tests/test_live_kg_graph.py
----------------------------
live_kg 的本体校验、种子文件读取、图谱构建（去重 / 剪枝 / 导出）。
"""

import pytest

from live_kg import ontology as ont
from live_kg.graph import CompanyGraph
from live_kg.seed import load_seed_relations


# ── 本体 ────────────────────────────────────────────────────────

def test_validate_edge_accepts_declared_types():
    assert ont.validate_edge("SUPPLIES_TO", ont.COMPANY, ont.COMPANY) is None
    assert ont.validate_edge("AFFECTS", ont.EVENT, ont.COMPANY) is None


@pytest.mark.parametrize("rel, src, dst", [
    ("SUPPLIES_TO", ont.COMPANY, ont.INDUSTRY),
    ("AFFECTS", ont.COMPANY, ont.EVENT),          # 方向反了
    ("REPORTED_BY", ont.EVENT, ont.COMPANY),
    ("OWNS", ont.COMPANY, ont.COMPANY),           # 未定义的关系
])
def test_validate_edge_rejects_violations(rel, src, dst):
    assert ont.validate_edge(rel, src, dst) is not None


def test_node_id_rejects_unknown_type():
    with pytest.raises(ValueError):
        ont.node_id("Person", "tim-cook")


def test_seed_relations_are_subset_of_ontology():
    assert ont.SEED_RELATIONS <= set(ont.RELATIONS)


# ── 种子文件 ────────────────────────────────────────────────────

def test_project_seed_file_is_valid():
    rows, errors = load_seed_relations()
    assert errors == []
    assert len(rows) >= 20
    assert {"subject": "TSM", "relation": "SUPPLIES_TO", "object": "AAPL"}.items() <= rows[0].items()


def test_seed_loader_reports_bad_rows_and_dedupes(tmp_path):
    f = tmp_path / "seed.csv"
    f.write_text(
        "subject,relation,object,note\n"
        "# 注释行\n"
        "tsm,supplies_to,aapl,小写也可以\n"
        "TSM,SUPPLIES_TO,AAPL,重复行\n"
        "AAPL,OWNS,BEATS,未定义关系\n"
        "apple inc,COMPETES_WITH,GOOGL,ticker 不合法\n"
        "MSFT,COMPETES_WITH,MSFT,自环\n"
        "\n"
        "NVDA,COMPETES_WITH,AMD,GPU\n",
        encoding="utf-8",
    )
    rows, errors = load_seed_relations(f)

    assert [(r["subject"], r["relation"], r["object"]) for r in rows] == [
        ("TSM", "SUPPLIES_TO", "AAPL"), ("NVDA", "COMPETES_WITH", "AMD"),
    ]
    assert rows[1]["line"] == 9
    assert len(errors) == 3
    assert errors[0].startswith("第 5 行")


def test_seed_loader_missing_file(tmp_path):
    rows, errors = load_seed_relations(tmp_path / "nope.csv")
    assert rows == [] and "不存在" in errors[0]


# ── 图谱构建 ────────────────────────────────────────────────────

@pytest.fixture
def g():
    graph = CompanyGraph()
    graph.load_seed([
        {"subject": "TSM", "relation": "SUPPLIES_TO", "object": "AAPL", "note": "代工"},
        {"subject": "AAPL", "relation": "COMPETES_WITH", "object": "GOOGL", "note": ""},
    ])
    return graph


def test_load_seed_creates_companies_and_edges(g):
    stats = g.stats()
    assert stats["nodes_by_type"] == {"Company": 3}
    assert stats["edges_by_relation"] == {"SUPPLIES_TO": 1, "COMPETES_WITH": 1}
    assert g.g.edges["company:TSM", "company:AAPL", "SUPPLIES_TO"]["source"] == "seed"


def test_add_company_merges_attributes(g):
    g.add_company("aapl", name="Apple Inc.", is_target=True)
    g.add_company("AAPL")                         # 再次添加不应覆盖 is_target
    node = g.g.nodes["company:AAPL"]
    assert node["name"] == "Apple Inc." and node["is_target"] is True
    assert g.stats()["nodes_by_type"]["Company"] == 3


def test_symmetric_relation_stored_once(g):
    assert g.add_company_relation("GOOGL", "COMPETES_WITH", "AAPL") is True
    assert g.stats()["edges_by_relation"]["COMPETES_WITH"] == 1


def test_invalid_edge_is_rejected_and_recorded(g):
    ind = g.add_industry("Consumer Electronics")
    assert g.add_relation("SUPPLIES_TO", "company:TSM", ind) is False
    assert g.add_relation("IN_INDUSTRY", "company:TSM", "industry:missing") is False
    assert len(g.rejected) == 2
    assert g.add_relation("IN_INDUSTRY", "company:AAPL", ind) is True
    assert ind == "industry:consumer-electronics"


def test_node_type_conflict_raises(g):
    g.g.add_node("news:abc", type="Event")
    with pytest.raises(ValueError):
        g._add_node(ont.NEWS, "abc")


def test_add_event_with_evidence_and_events_for(g):
    news = g.add_news("TSMC capacity constrained", "Bloomberg", "2026-09-12T10:00", "https://n/1")
    eid = g.add_event("SUPPLY_CHAIN", "negative", "2026-09-12", "产能受限", ["tsm"], [news], 0.8)

    assert eid == "event:E1"
    events = g.events_for("TSM")
    assert events[0]["id"] == "E1"
    assert events[0]["sources"][0]["url"] == "https://n/1"
    assert g.event_ids() == {"E1"}


def test_duplicate_event_is_merged_and_evidence_accumulated(g):
    n1 = g.add_news("A", url="https://n/1")
    n2 = g.add_news("B", url="https://n/2")
    e1 = g.add_event("SUPPLY_CHAIN", "negative", "2026-09-12", "台积电 产能受限", ["TSM"], [n1], 0.6)
    e2 = g.add_event("SUPPLY_CHAIN", "negative", "2026-09-12", "台积电产能受限", ["TSM"], [n2], 0.9)

    assert e1 == e2
    ev = g.events_for("TSM")
    assert len(ev) == 1
    assert ev[0]["confidence"] == 0.9
    assert {s["url"] for s in ev[0]["sources"]} == {"https://n/1", "https://n/2"}


def test_same_news_url_is_one_node(g):
    assert g.add_news("Title", url="https://n/1") == g.add_news("Title (updated)", url="https://n/1")


@pytest.mark.parametrize("kwargs", [
    {"event_type": "RUMOR"},
    {"polarity": "very_bad"},
    {"affected": ["ZZZZ"]},
])
def test_invalid_event_is_rejected(g, kwargs):
    base = dict(event_type="PRODUCT", polarity="positive", date="2026-09-10",
                summary="s", affected=["AAPL"], evidence=[], confidence=0.5)
    base.update(kwargs)
    assert g.add_event(**base) is None
    assert g.rejected


def test_event_confidence_is_clamped(g):
    g.add_event("PRODUCT", "positive", "2026-09-10", "s", ["AAPL"], [], confidence=3)
    assert g.events_for("AAPL")[0]["confidence"] == 1.0


def test_prune_companies_removes_dangling_events_and_news(g):
    news = g.add_news("Google news", url="https://n/g")
    g.add_event("PRODUCT", "positive", "2026-09-10", "Pixel", ["GOOGL"], [news], 0.7)
    g.add_event("PRODUCT", "positive", "2026-09-10", "iPhone", ["AAPL"], [], 0.7)

    g.prune_companies({"AAPL", "TSM"})

    assert not g.has_company("GOOGL")
    assert g.event_ids() == {"E2"}
    assert g.stats()["nodes_by_type"] == {"Company": 2, "Event": 1}


def test_to_dict_is_json_serializable(g):
    import json
    g.add_event("PRODUCT", "positive", "2026-09-10", "iPhone", ["AAPL"], [], 0.7)
    data = json.loads(json.dumps(g.to_dict(), ensure_ascii=False))
    assert {n["id"] for n in data["nodes"]} >= {"company:AAPL", "event:E1"}
    assert any(e["relation"] == "AFFECTS" for e in data["edges"])
    assert data["stats"]["edge_count"] == 3


def test_to_mermaid_contains_nodes_edges_and_classes(g):
    g.add_company("AAPL", name='Apple "Inc."', is_target=True)
    g.add_relation("IN_INDUSTRY", "company:AAPL", g.add_industry("Consumer Electronics"))
    for i in range(5):
        g.add_event("PRODUCT", "negative", f"2026-09-0{i+1}", f"事件{i}", ["TSM"], [], 0.1 * (i + 1))

    md = g.to_mermaid(max_events_per_company=2, highlight_events={"E1"})

    assert md.startswith("graph LR")
    assert "Apple 'Inc.' (AAPL)\"]:::target" in md          # 双引号被转义
    assert "-->|SUPPLIES_TO|" in md and "<-->|COMPETES_WITH|" in md and "-->|IN_INDUSTRY|" in md
    events_drawn = [l for l in md.splitlines() if ":::negative" in l]
    # 高亮的 E1 必画；另一个是置信度最高的 E5
    assert len(events_drawn) == 2
    assert any("E1 " in l for l in events_drawn) and any("E5 " in l for l in events_drawn)
    assert "classDef negative" in md
