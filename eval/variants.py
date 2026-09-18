"""
eval/variants.py
-----------------
三种实验配置（见 docs/eval-design.md 第 2 节）：

  A 基线      只有目标公司的 4 个数据工具；prompt 不含任何产业链信息
  B 纯 toolu  工具同 A，但允许查询 universe 中任意公司；prompt 里以**文字**给出
              与 C 完全相同的种子关系，由模型自己决定查谁、如何推理传导
  C 图谱版    当前生产配置（4 个数据工具 + query_company_graph）

B 与 C 拿到的关系信息完全一致，差别只在于"文本交给模型自由使用"还是
"结构化图谱 + 确定性规则"，因此 B↔C 的差距才能归因到设计本身。

emit_report 字段按变体裁剪：A 无 supply_chain_analysis / cited_event_ids，
B 有 supply_chain_analysis、无 cited_event_ids，C 两者都有。
"""

from __future__ import annotations

import copy
from typing import Optional

import orchestrator_loop as ol
from live_kg.queries import ROLE_ZH, find_neighbors
from orchestrator import OrchestratorAgent

VARIANTS = ("A", "B", "C")

VARIANT_DESC = {
    "A": "基线：只看目标公司自身数据",
    "B": "纯 tool use：种子关系以文本给出，模型自行查询相关公司新闻",
    "C": "知识图谱：结构化图谱 + 规则风险传导",
}

# ── emit_report 字段裁剪 ─────────────────────────────────────────

_GRAPH_ONLY_FIELDS = {"A": ["supply_chain_analysis", "cited_event_ids"], "B": ["cited_event_ids"], "C": []}


def _emit_tool_for(variant: str) -> dict:
    emit = copy.deepcopy(next(t for t in ol.TOOL_DEFINITIONS if t["name"] == "emit_report"))
    for field in _GRAPH_ONLY_FIELDS[variant]:
        emit["input_schema"]["properties"].pop(field, None)
        if field in emit["input_schema"]["required"]:
            emit["input_schema"]["required"].remove(field)
    if variant == "B":
        emit["input_schema"]["properties"]["supply_chain_analysis"]["description"] = (
            "产业链与竞争格局分析：结合你查询到的相关公司新闻，说明可能的风险传导"
        )
    return emit


def tool_definitions(variant: str) -> list[dict]:
    """A/B：4 个数据工具 + emit_report；C：再加 query_company_graph"""
    keep = {"query_market_data", "query_news", "query_sec_filings", "query_macro"}
    if variant == "C":
        keep.add("query_company_graph")
    tools = [copy.deepcopy(t) for t in ol.TOOL_DEFINITIONS if t["name"] in keep]
    if variant in ("A", "B"):
        for t in tools:
            if t["name"] in ("query_market_data", "query_news"):
                t["input_schema"]["properties"]["entity"]["description"] = (
                    "公司名或股票代码（只能查询目标公司）" if variant == "A"
                    else "公司名或股票代码（可以是目标公司，也可以是下方列出的相关公司）"
                )
    return tools + [_emit_tool_for(variant)]


# ── prompt ───────────────────────────────────────────────────────

_BASE_PROMPT = """你是一位专业的金融研究员 Agent。
你的任务是基于最新的真实数据，为指定公司生成投资建议报告。

你有以下工具可以使用：
- query_market_data：实时公司概况、估值/财务指标、近期行情与技术指标
- query_news：近两周的公司新闻
- query_sec_filings：近一年的 SEC 监管申报（10-K / 10-Q / 8-K 等）
- query_macro：当前宏观经济指标（利率、通胀、失业率、VIX）
{extra_tools}
工作流程建议（你可以根据情况调整，可以在同一轮并行调用多个工具）：
1. 查询市场数据，掌握基本面、估值与技术面
2. 查询近期新闻，自行判断新闻情绪，并关注重大事件
3. 查询 SEC 申报，确认近期财报 / 重大事项披露
4. 查询宏观指标，判断整体市场环境
{extra_steps}
注意：
- 只使用工具返回的数据，引用数值时注明数据截至日期；不要编造数字、新闻或申报
- 某个工具返回 error 时，不要反复重试同一调用；在报告对应部分写明"数据不可用"及原因，并相应降低置信度
- 宏观信号决定整体仓位方向，公司基本面/技术面/新闻决定个股判断
- 若各类信号方向相反（如基本面强但技术面走弱、宏观偏空），在报告中明确标注冲突并说明如何取舍
- emit_report 的 confidence 需真实反映数据完整度与信号一致性，不要过度自信
{extra_notes}- emit_report 每个文本字段控制在 150 字以内，key_signals 3-5 条，保持简洁"""

_B_RELATION_INTRO = """
【目标公司的产业链与竞争关系】（人工整理的公开信息）
{relations}
你可以对上面这些公司调用 query_news / query_market_data，判断它们近期的事件会不会传导到目标公司
（例如供应商的负面事件可能影响供应，竞争对手的利好可能带来竞争压力）。
"""


def relations_text(target: str, graph) -> str:
    """把种子关系写成文字，给 B 组使用；内容与 C 组图谱的邻居完全一致"""
    neighbors = find_neighbors(graph, target)
    if not neighbors:
        return f"（{target} 在已整理的关系数据中没有相关公司）"

    by_role: dict[str, list[str]] = {}
    for n in neighbors:
        for role in n["roles"]:
            label = n["ticker"] if n["hop"] == 1 else f"{n['ticker']}（经由 {n['via']}）"
            by_role.setdefault(ROLE_ZH[role], []).append(label)

    order = ["供应商", "客户", "合作伙伴", "竞争对手", "上游供应商"]
    lines = [f"- {role}：{'、'.join(dict.fromkeys(by_role[role]))}"
             for role in order if role in by_role]
    return f"{target} 的相关公司：\n" + "\n".join(lines)


def system_prompt(variant: str, target: str, graph=None) -> str:
    if variant == "A":
        return _BASE_PROMPT.format(
            extra_tools="", extra_steps="5. 信息足够时，调用 emit_report 输出结构化报告\n",
            extra_notes="")
    if variant == "B":
        return _BASE_PROMPT.format(
            extra_tools="",
            extra_steps="5. 结合下方的关系信息，查询你认为相关的公司，分析风险传导\n"
                        "6. 信息足够时，调用 emit_report 输出结构化报告\n",
            extra_notes="- 关系信息来自人工整理，传导分析属于推断，表述时注意分寸\n",
        ) + "\n" + _B_RELATION_INTRO.format(relations=relations_text(target, graph))
    return ol.SYSTEM_PROMPT      # C 组用生产 prompt


# ── 组装 Agent ───────────────────────────────────────────────────

def build_agent(variant: str, store, target: str, graph=None,
                temperature: float = 0, **agent_kwargs) -> OrchestratorAgent:
    """
    按变体组装一个从快照回放数据的 OrchestratorAgent。

    store : eval.snapshot.SnapshotStore
    graph : 种子关系图（B 组生成关系文本用），默认自行加载
    """
    if variant not in VARIANTS:
        raise ValueError(f"未知变体：{variant}（可选 {VARIANTS}）")
    if variant == "B" and graph is None:
        from eval.snapshot import seed_graph
        graph = seed_graph()

    agent = OrchestratorAgent(
        tool_definitions=tool_definitions(variant),
        system_prompt=system_prompt(variant, target.upper(), graph),
        temperature=temperature,
        **agent_kwargs,
    )
    # 图谱工具需要用到 loop 自带的 extractor，因此构造完 agent 再注入实现
    agent.loop.tool_impls = store.tool_impls(
        target=target.upper(),
        restrict_to=target.upper() if variant == "A" else None,
        extractor=agent.loop.extractor if variant == "C" else None,
    )
    return agent
