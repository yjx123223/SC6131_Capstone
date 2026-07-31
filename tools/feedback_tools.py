"""
tools/feedback_tools.py
------------------------
get_feedback_stats 工具的唯一实现。

之前这段逻辑在 orchestrator_loop.py（Orchestrator 的工具分发）和
mcp_servers/kg_server.py 的 get_feedback_stats 里各写了一份，
现在合并到这里，两边都改为调用本函数。
"""

from . import resources


def get_feedback_stats(relation_type: str = "", store=None) -> dict:
    """
    获取历史建议的用户评分统计，了解哪类 KG 信号关系类型
    在过去的建议中表现更好（平均评分更高）。

    Parameters
    ----------
    relation_type : 按关系类型过滤（子串匹配，大小写不敏感），留空返回全部
    store         : 已有的 FeedbackStore 实例；不传则使用 resources 共享单例

    Returns
    -------
    成功：{"total_rated": int, "positive_rate": float, "signal_stats": dict}
    失败：{"error": str}
    """
    store = store or resources.get_store()

    try:
        report = store.signal_accuracy_report()
    except Exception as e:
        return {"error": f"评分查询失败：{e}"}

    stats = report.get("signal_stats", {})
    if relation_type:
        stats = {k: v for k, v in stats.items() if relation_type.lower() in k.lower()}

    return {
        "total_rated":   report.get("total_rated", 0),
        "positive_rate": report.get("positive_rate", 0),
        "signal_stats":  stats,
    }
