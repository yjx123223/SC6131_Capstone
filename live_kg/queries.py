"""
live_kg/queries.py
-------------------
图谱查询与推理（确定性规则，不调用 LLM）：

  find_neighbors()      从目标公司出发找 1 跳邻居，再沿供应链找 2 跳（供应商的供应商）
  propagate_risks()     根据"邻居与目标的关系 × 邻居事件的正负面"推导对目标公司的影响

推理规则（RULES）：
  邻居是目标的供应商，发生负面事件 → 供应风险      权重 1.0
  邻居是目标的客户，  发生负面事件 → 需求风险      权重 0.8
  邻居是目标的合作方，发生负面事件 → 合作风险      权重 0.6
  邻居是目标的竞争者，发生正面事件 → 竞争压力      权重 0.6
  邻居是目标的竞争者，发生负面事件 → 对手承压（可能利好，单独列为 opportunity）
  2 跳的供应商的供应商按"供应商"规则计算，再乘跳数衰减（config.KG_HOP_DECAY）

分数 = 关系权重 × 跳数衰减 × 事件置信度
"""

from typing import Optional

import config
from .graph import CompanyGraph

# 邻居角色的优先级（1 跳数量超过上限时按此顺序截断）
ROLE_PRIORITY = ["supplier", "customer", "partner", "competitor"]
ROLE_ZH = {
    "supplier": "供应商",
    "customer": "客户",
    "partner": "合作伙伴",
    "competitor": "竞争对手",
    "supplier_of_supplier": "上游供应商",
}

# (角色, 事件正负面) → (影响标签, 权重, 类别)
RULES = {
    ("supplier", "negative"):   ("供应风险", 1.0, "risk"),
    ("customer", "negative"):   ("需求风险", 0.8, "risk"),
    ("partner", "negative"):    ("合作风险", 0.6, "risk"),
    ("competitor", "positive"): ("竞争压力", 0.6, "risk"),
    ("competitor", "negative"): ("竞争对手承压", 0.6, "opportunity"),
}


def _roles_toward(graph: CompanyGraph, target: str, other: str) -> list[str]:
    roles = set()
    for s, rel, o in graph.company_edges(target):
        if other not in (s, o):
            continue
        if rel == "SUPPLIES_TO":
            roles.add("supplier" if s == other else "customer")
        elif rel == "COMPETES_WITH":
            roles.add("competitor")
        elif rel == "PARTNERS_WITH":
            roles.add("partner")
    return [r for r in ROLE_PRIORITY if r in roles]


def _suppliers_of(graph: CompanyGraph, ticker: str) -> list[str]:
    return sorted({s for s, rel, o in graph.company_edges(ticker) if rel == "SUPPLIES_TO" and o == ticker})


def find_neighbors(
    graph: CompanyGraph,
    target: str,
    max_hop1: Optional[int] = None,
    max_hop2: Optional[int] = None,
) -> list[dict]:
    """
    Returns
    -------
    [
      {"ticker", "hop": 1|2, "roles": [...], "via": str|None,
       "path": [(subject, relation, object), ...]},
      ...
    ]
    1 跳按角色优先级 + ticker 字母序排序后截断；2 跳只沿 1 跳供应商向上游找。
    """
    target = target.upper()
    max_hop1 = config.KG_MAX_HOP1 if max_hop1 is None else max_hop1
    max_hop2 = config.KG_MAX_HOP2 if max_hop2 is None else max_hop2

    others = {s if s != target else o for s, _, o in graph.company_edges(target)}
    hop1 = []
    for other in others:
        roles = _roles_toward(graph, target, other)
        if not roles:
            continue
        path = [e for e in graph.company_edges(target) if other in (e[0], e[2])]
        hop1.append({"ticker": other, "hop": 1, "roles": roles, "via": None, "path": sorted(path)})
    hop1.sort(key=lambda n: (ROLE_PRIORITY.index(n["roles"][0]), n["ticker"]))
    hop1 = hop1[:max_hop1]

    chosen = {target} | {n["ticker"] for n in hop1}
    hop2 = []
    for n in hop1:
        if "supplier" not in n["roles"]:
            continue
        for upstream in _suppliers_of(graph, n["ticker"]):
            if upstream in chosen or len(hop2) >= max_hop2:
                continue
            chosen.add(upstream)
            hop2.append({
                "ticker": upstream,
                "hop": 2,
                "roles": ["supplier_of_supplier"],
                "via": n["ticker"],
                "path": [(upstream, "SUPPLIES_TO", n["ticker"]), (n["ticker"], "SUPPLIES_TO", target)],
            })
    return hop1 + hop2


_ROLE_RELATION = {"supplier": "SUPPLIES_TO", "customer": "SUPPLIES_TO",
                  "competitor": "COMPETES_WITH", "partner": "PARTNERS_WITH"}


def _path_for_role(neighbor: dict, role: str, target: str) -> list[tuple[str, str, str]]:
    """从邻居的所有边里挑出与该角色对应的那条（2 跳直接用完整链路）"""
    if role == "supplier_of_supplier":
        return neighbor["path"]
    rel = _ROLE_RELATION[role]
    other = neighbor["ticker"]
    for s, r, o in neighbor["path"]:
        if r != rel:
            continue
        if role == "supplier" and (s, o) != (other, target):
            continue
        if role == "customer" and (s, o) != (target, other):
            continue
        return [(s, r, o)]
    return neighbor["path"]


def path_text(path: list[tuple[str, str, str]]) -> str:
    """
    首尾相接的链路合并成一条：[(ASML, SUPPLIES_TO, TSM), (TSM, SUPPLIES_TO, AAPL)]
      → 'ASML ─SUPPLIES_TO→ TSM ─SUPPLIES_TO→ AAPL'
    不相接的多条边用"；"分隔。
    """
    if not path:
        return ""
    chained = all(path[i][2] == path[i + 1][0] for i in range(len(path) - 1))
    if chained:
        text = path[0][0]
        for _, r, o in path:
            text += f" ─{r}→ {o}"
        return text
    return "；".join(f"{s} ─{r}→ {o}" for s, r, o in path)


def propagate_risks(graph: CompanyGraph, target: str, neighbors: list[dict]) -> dict:
    """
    Returns
    -------
    {
      "risks":         [...按 score 降序],
      "opportunities": [...按 score 降序],
    }
    每项：{event_id, neighbor, hop, role, role_zh, impact, score, polarity,
           event_type, date, summary, confidence, path, sources}
    """
    target = target.upper()
    risks, opportunities = [], []
    for n in neighbors:
        decay = config.KG_HOP_DECAY.get(n["hop"], 0.0)
        events = graph.events_for(n["ticker"])
        for role in n["roles"]:
            rule_role = "supplier" if role == "supplier_of_supplier" else role
            for ev in events:
                rule = RULES.get((rule_role, ev["polarity"]))
                if rule is None:
                    continue
                impact, weight, kind = rule
                if role == "supplier_of_supplier":
                    impact = "上游" + impact
                item = {
                    "event_id":   ev["id"],
                    "neighbor":   n["ticker"],
                    "hop":        n["hop"],
                    "role":       role,
                    "role_zh":    ROLE_ZH[role],
                    "impact":     impact,
                    "score":      round(weight * decay * ev["confidence"], 3),
                    "polarity":   ev["polarity"],
                    "event_type": ev["event_type"],
                    "date":       ev["date"],
                    "summary":    ev["summary"],
                    "confidence": ev["confidence"],
                    "path":       path_text(_path_for_role(n, role, target)),
                    "sources":    ev["sources"],
                }
                (risks if kind == "risk" else opportunities).append(item)

    key = lambda x: (-x["score"], x["hop"], x["event_id"])
    return {"risks": sorted(risks, key=key), "opportunities": sorted(opportunities, key=key)}
