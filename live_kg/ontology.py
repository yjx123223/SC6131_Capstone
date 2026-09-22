"""
live_kg/ontology.py
--------------------
本体定义：规定图谱里"能有哪些类型的节点、哪些关系、关系两端必须是什么类型"。

类比：本体是"类的定义"，CompanyGraph 里的节点和边是"new 出来的对象"。
所有写入图谱的边都要经过 validate_edge() 校验。
"""

from typing import Optional

# ── 实体类型 ─────────────────────────────────────────────────────
COMPANY = "Company"
INDUSTRY = "Industry"
EVENT = "Event"
NEWS = "NewsArticle"
FILING = "Filing"

NODE_TYPES = {COMPANY, INDUSTRY, EVENT, NEWS, FILING}

# 节点 ID 前缀（node_id = "<prefix>:<key>"）
NODE_PREFIX = {
    COMPANY:  "company",
    INDUSTRY: "industry",
    EVENT:    "event",
    NEWS:     "news",
    FILING:   "filing",
}

# ── 关系 ─────────────────────────────────────────────────────────
# 关系名 → (起点类型, 终点类型, 是否对称, 中文说明)
RELATIONS = {
    "IN_INDUSTRY":   (COMPANY, INDUSTRY, False, "所属行业"),
    "FILED":         (COMPANY, FILING,   False, "提交申报"),
    "SUPPLIES_TO":   (COMPANY, COMPANY,  False, "供应给"),
    "COMPETES_WITH": (COMPANY, COMPANY,  True,  "竞争"),
    "PARTNERS_WITH": (COMPANY, COMPANY,  True,  "合作"),
    "AFFECTS":       (EVENT,   COMPANY,  False, "影响"),
    "REPORTED_BY":   (EVENT,   NEWS,     False, "报道来源"),
}

# 允许出现在人工种子文件中的关系
SEED_RELATIONS = {"SUPPLIES_TO", "COMPETES_WITH", "PARTNERS_WITH"}

# ── 事件类型与正负面 ──────────────────────────────────────────────
EVENT_TYPES = {
    "EARNINGS":       "财报",
    "GUIDANCE":       "业绩指引",
    "PRODUCT":        "产品",
    "M_AND_A":        "并购",
    "PARTNERSHIP":    "合作",
    "REGULATORY":     "监管",
    "LEGAL":          "诉讼",
    "MANAGEMENT":     "管理层变动",
    "SUPPLY_CHAIN":   "供应链",
    "ANALYST_RATING": "分析师评级",
    "MACRO":          "宏观",
    "OTHER":          "其他",       # 本体未覆盖的情况，用于迭代本体
}

POLARITIES = ("positive", "negative", "neutral")
POLARITY_ZH = {"positive": "正面", "negative": "负面", "neutral": "中性"}


def node_id(node_type: str, key: str) -> str:
    if node_type not in NODE_TYPES:
        raise ValueError(f"未知实体类型：{node_type}")
    return f"{NODE_PREFIX[node_type]}:{key}"


def is_symmetric(relation: str) -> bool:
    return RELATIONS[relation][2]


def validate_edge(relation: str, src_type: str, dst_type: str) -> Optional[str]:
    """合法返回 None，不合法返回原因"""
    if relation not in RELATIONS:
        return f"未知关系：{relation}"
    expected_src, expected_dst, _, _ = RELATIONS[relation]
    if src_type != expected_src or dst_type != expected_dst:
        return (
            f"{relation} 要求 {expected_src} → {expected_dst}，"
            f"实际为 {src_type} → {dst_type}"
        )
    return None
