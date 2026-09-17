"""
tests/test_live_kg_queries.py
------------------------------
live_kg.queries：邻居发现（1~2 跳、角色优先级截断）与风险传导推理规则。
"""

import pytest

from live_kg.graph import CompanyGraph
from live_kg.queries import find_neighbors, path_text, propagate_risks

SEED = [
    ("TSM", "SUPPLIES_TO", "AAPL"),
    ("QCOM", "SUPPLIES_TO", "AAPL"),
    ("ASML", "SUPPLIES_TO", "TSM"),
    ("TSM", "SUPPLIES_TO", "QCOM"),        # QCOM 的上游也是 TSM，不应重复加入
    ("AMAT", "SUPPLIES_TO", "TSM"),
    ("AAPL", "SUPPLIES_TO", "BBY"),        # AAPL 的客户
    ("AAPL", "COMPETES_WITH", "GOOGL"),
    ("AAPL", "PARTNERS_WITH", "GOOGL"),
    ("MSFT", "COMPETES_WITH", "AAPL"),
    ("NVDA", "SUPPLIES_TO", "MSFT"),       # 与 AAPL 无直接关系
]


@pytest.fixture
def g():
    graph = CompanyGraph()
    graph.load_seed([{"subject": s, "relation": r, "object": o} for s, r, o in SEED])
    return graph


def _event(g, ticker, polarity, confidence=1.0, date="2026-09-10", etype="OTHER"):
    return g.add_event(etype, polarity, date, f"{ticker} {polarity} {date}", [ticker], [], confidence)


# ── 邻居发现 ────────────────────────────────────────────────────

def test_hop1_roles_and_priority_order(g):
    nb = find_neighbors(g, "aapl", max_hop1=10, max_hop2=0)
    assert [(n["ticker"], n["roles"]) for n in nb] == [
        ("QCOM", ["supplier"]),
        ("TSM", ["supplier"]),
        ("BBY", ["customer"]),
        ("GOOGL", ["partner", "competitor"]),
        ("MSFT", ["competitor"]),
    ]
    assert all(n["hop"] == 1 and n["via"] is None for n in nb)


def test_hop1_is_truncated_by_role_priority(g):
    nb = find_neighbors(g, "AAPL", max_hop1=2, max_hop2=0)
    assert [n["ticker"] for n in nb] == ["QCOM", "TSM"]


def test_hop2_follows_supply_chain_only_and_skips_known(g):
    nb = find_neighbors(g, "AAPL", max_hop1=10, max_hop2=10)
    hop2 = [n for n in nb if n["hop"] == 2]
    # QCOM 的上游 TSM 已经是 1 跳，不重复；TSM 的上游 AMAT、ASML
    assert [(n["ticker"], n["via"]) for n in hop2] == [("AMAT", "TSM"), ("ASML", "TSM")]
    assert hop2[1]["path"] == [("ASML", "SUPPLIES_TO", "TSM"), ("TSM", "SUPPLIES_TO", "AAPL")]
    assert "NVDA" not in {n["ticker"] for n in nb}


def test_hop2_limit(g):
    nb = find_neighbors(g, "AAPL", max_hop1=10, max_hop2=1)
    assert [n["ticker"] for n in nb if n["hop"] == 2] == ["AMAT"]


def test_unknown_or_isolated_company_has_no_neighbors(g):
    assert find_neighbors(g, "ZZZZ") == []
    g.add_company("IBM")
    assert find_neighbors(g, "IBM") == []


def test_path_text():
    assert path_text([("ASML", "SUPPLIES_TO", "TSM"), ("TSM", "SUPPLIES_TO", "AAPL")]) == \
        "ASML ─SUPPLIES_TO→ TSM ─SUPPLIES_TO→ AAPL"
    assert path_text([("AAPL", "COMPETES_WITH", "GOOGL"), ("AAPL", "PARTNERS_WITH", "GOOGL")]) == \
        "AAPL ─COMPETES_WITH→ GOOGL；AAPL ─PARTNERS_WITH→ GOOGL"
    assert path_text([]) == ""


# ── 风险传导 ────────────────────────────────────────────────────

def _run(g, **kw):
    nb = find_neighbors(g, "AAPL", max_hop1=10, max_hop2=10)
    return propagate_risks(g, "AAPL", nb)


def test_supplier_negative_event_is_supply_risk(g):
    _event(g, "TSM", "negative", confidence=0.8)
    risks = _run(g)["risks"]
    assert len(risks) == 1
    r = risks[0]
    assert (r["impact"], r["role"], r["score"], r["hop"]) == ("供应风险", "supplier", 0.8, 1)
    assert r["path"] == "TSM ─SUPPLIES_TO→ AAPL"


def test_customer_negative_event_is_demand_risk(g):
    _event(g, "BBY", "negative")
    r = _run(g)["risks"][0]
    assert (r["impact"], r["score"], r["path"]) == ("需求风险", 0.8, "AAPL ─SUPPLIES_TO→ BBY")


def test_second_hop_is_decayed(g):
    _event(g, "ASML", "negative", confidence=0.9)
    r = _run(g)["risks"][0]
    assert r["impact"] == "上游供应风险"
    assert r["hop"] == 2
    assert r["score"] == pytest.approx(0.45)
    assert r["path"] == "ASML ─SUPPLIES_TO→ TSM ─SUPPLIES_TO→ AAPL"


def test_competitor_positive_is_risk_negative_is_opportunity(g):
    _event(g, "MSFT", "positive", date="2026-09-10")
    _event(g, "MSFT", "negative", date="2026-09-11")
    result = _run(g)
    assert [(x["impact"], x["neighbor"]) for x in result["risks"]] == [("竞争压力", "MSFT")]
    assert [(x["impact"], x["neighbor"]) for x in result["opportunities"]] == [("竞争对手承压", "MSFT")]


def test_multi_role_neighbor_yields_item_per_role(g):
    _event(g, "GOOGL", "negative")
    result = _run(g)
    assert [(x["impact"], x["path"]) for x in result["risks"]] == [("合作风险", "AAPL ─PARTNERS_WITH→ GOOGL")]
    assert [x["impact"] for x in result["opportunities"]] == ["竞争对手承压"]


@pytest.mark.parametrize("ticker, polarity", [
    ("TSM", "positive"),     # 供应商利好：不算风险
    ("TSM", "neutral"),
    ("AAPL", "negative"),    # 目标自身事件不参与传导
    ("NVDA", "negative"),    # 非邻居
])
def test_events_without_rule_are_ignored(g, ticker, polarity):
    _event(g, ticker, polarity)
    result = _run(g)
    assert result == {"risks": [], "opportunities": []}


def test_risks_sorted_by_score_and_include_sources(g):
    news = g.add_news("TSMC", "Bloomberg", "2026-09-12T10:00", "https://n/1")
    g.add_event("SUPPLY_CHAIN", "negative", "2026-09-12", "产能受限", ["TSM"], [news], 0.5)
    _event(g, "QCOM", "negative", confidence=0.9)
    _event(g, "ASML", "negative", confidence=1.0)

    risks = _run(g)["risks"]
    assert [x["neighbor"] for x in risks] == ["QCOM", "TSM", "ASML"]   # 0.9 > 0.5 = 0.5(2 跳), 同分时 1 跳优先
    assert risks[1]["sources"][0]["url"] == "https://n/1"
    assert risks[1]["event_type"] == "SUPPLY_CHAIN"
