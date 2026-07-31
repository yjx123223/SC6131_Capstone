"""
tools/resources.py
-------------------
共享的重资源懒加载单例：FinDKGGraph / FeedbackStore。

背景：mcp_servers/kg_server.py 之前自己维护了一份懒加载单例。
现在把它收敛到这里，作为"没有自带实例的调用方"的默认资源提供方。

orchestrator.py 的实例是从 main.py 通过参数传入的（避免在同一进程里
重复加载一次 FinDKG 数据集），因此不强制经过这里——kg_tools /
feedback_tools 的函数都允许显式传入已有实例，只有不传时才回退到
本模块的单例。

注：早期版本这里还有 get_predictor()（KGPredictor 嵌入相似度预测），
已随 Multi-Agent 主链路摘除 predictor 一并移除。单 Agent 链路
（llm_advisor.py / main.py 的 _init_components）自行构造 KGPredictor，
不依赖本模块。
"""

from typing import Optional

_graph = None
_store = None


def get_graph():
    """返回共享的 FinDKGGraph 单例（首次调用时加载数据集）"""
    global _graph
    if _graph is None:
        from kg_query import FinDKGGraph
        _graph = FinDKGGraph()
    return _graph


def get_store(db_path: Optional[str] = None):
    """返回共享的 FeedbackStore 单例"""
    global _store
    if _store is None:
        from feedback_store import FeedbackStore
        _store = FeedbackStore(db_path=db_path)
    return _store


def reset():
    """
    重置所有单例缓存。

    主要给测试用：每个测试用例需要一个干净的 FeedbackStore（比如指向
    临时数据库文件）时，先 reset() 再调用 get_store(db_path=...)。
    """
    global _graph, _store
    _graph = None
    _store = None
