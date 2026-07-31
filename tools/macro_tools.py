"""
tools/macro_tools.py
---------------------
query_macro 工具的唯一实现。

之前这段逻辑在 orchestrator_loop.py（Orchestrator 的工具分发）和
mcp_servers/macro_server.py 的 query_macro 里各写了一份，且两份
实现有细微差异（macro_server 版本多返回 indicator_docs 字段）。
合并时采用 macro_server 这份更完整的输出。

fred_api_key 作为显式参数传入，不在函数内部读环境变量——
调用方（orchestrator / macro_server）各自负责从环境变量或配置中
取得 key 后传进来，这样这个函数本身不依赖全局状态，测试时也方便 mock。
"""

from typing import Optional


# 供 LLM / 报告展示参考的指标说明
_INDICATOR_DOCS = {
    "fed_funds_rate": "联邦基金利率（%），反映货币政策松紧",
    "treasury_10y":   "10年期国债收益率（%），反映长端利率预期",
    "cpi_yoy":        "CPI 同比变化（%），反映通货膨胀水平",
    "unemployment":   "失业率（%），反映劳动力市场状况",
    "vix":            "VIX 恐慌指数，反映市场隐含波动率和风险偏好",
}


def query_macro(fred_api_key: Optional[str], indicators: Optional[list[str]] = None) -> dict:
    """
    从 FRED（美联储经济数据库）获取最新宏观经济指标。

    Parameters
    ----------
    fred_api_key : FRED API Key，未配置时返回 error
    indicators   : 需要查询的指标列表，可选值：
                   fed_funds_rate, treasury_10y, cpi_yoy, unemployment, vix
                   不传则返回全部指标

    Returns
    -------
    成功：{"indicators": dict, "summary_text": str, "indicator_docs": dict}
    失败：{"error": str}
    """
    if not fred_api_key:
        return {"error": "未配置 FRED_API_KEY，宏观数据不可用"}

    try:
        from macro_agent import _fetch_fred_indicators, _format_indicators
    except ImportError as e:
        return {"error": f"macro_agent 模块不可用：{e}"}

    try:
        all_indicators = _fetch_fred_indicators(fred_api_key)
    except Exception as e:
        return {"error": f"FRED 数据拉取失败：{e}"}

    requested = set(indicators or [])
    filtered = (
        {k: v for k, v in all_indicators.items() if k in requested}
        if requested else all_indicators
    )

    return {
        "indicators":     filtered,
        "summary_text":   _format_indicators(filtered),
        "indicator_docs": {k: _INDICATOR_DOCS[k] for k in filtered if k in _INDICATOR_DOCS},
    }
