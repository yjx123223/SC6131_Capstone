"""
tools/kg_tools.py
------------------
query_kg_signals 工具的唯一实现。

之前这段逻辑在 orchestrator_loop.py（Orchestrator 的工具分发）和
mcp_servers/kg_server.py 的 query_kg_signals 里各写了一份，
现在合并到这里，两边都改为调用本函数。

注：早期版本还支持附带 KGTransformer 嵌入相似度预测信号
（kg_predictor.KGPredictor），因预测质量不完善，已从 Multi-Agent
主链路移除——本函数只返回 KG 中的真实历史事件。单 Agent 链路
（llm_advisor.py）不走本函数，仍保留自己的 predictor 集成。
"""

from . import resources


def query_kg_signals(
    entity: str,
    weeks: int = 12,
    graph=None,
    include_multihop: bool = True,
) -> dict:
    """
    查询目标实体在 FinDKG 知识图谱中的历史事件信号，
    包括正面影响事件（Positive_Impact_On / Raise / Invests_In）、
    负面影响事件（Negative_Impact_On / Decrease）、其他关联事件，
    以及 2 跳 BFS 间接关联路径（可选）。

    Parameters
    ----------
    entity           : 实体名称，如 "Apple Inc."
    weeks            : 查询最近 N 周，默认 12
    graph            : 已有的 FinDKGGraph 实例；不传则使用 resources 共享单例
    include_multihop : 是否附带 2 跳间接关联路径，默认 True

    Returns
    -------
    成功：
        {
            "entity": str, "period": str, "total_events": int,
            "positive_impacts": [...], "negative_impacts": [...],
            "other_relations": [...],
            "multihop_context": str,   # 仅 include_multihop=True 时存在
        }
    失败：{"error": str}
    """
    graph = graph or resources.get_graph()

    summary = graph.get_impact_summary(entity, n_recent_weeks=weeks)

    if summary.get("total_events", 0) == 0:
        return {"error": f"KG 中未找到 '{entity}' 的近期数据"}

    def _fmt(events):
        return [
            {
                "subject":  e["subject"],
                "relation": e["relation"],
                "object":   e["object"],
                "count":    e["count"],
            }
            for e in events[:6]
        ]

    result = {
        "entity":           entity,
        "period":           summary.get("period", "未知"),
        "total_events":     summary.get("total_events", 0),
        "positive_impacts": _fmt(summary.get("positive_impacts", [])),
        "negative_impacts": _fmt(summary.get("negative_impacts", [])),
        "other_relations":  _fmt(summary.get("other_relations", [])),
    }

    if include_multihop:
        try:
            multihop = graph.get_multihop_context(
                entity, max_hops=2, weeks=weeks, max_width=10
            )
            result["multihop_context"] = graph.format_multihop_for_prompt(multihop)
        except Exception as e:
            result["multihop_context"] = f"（多跳查询失败：{e}）"

    return result
