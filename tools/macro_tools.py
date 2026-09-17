"""
tools/macro_tools.py
---------------------
query_macro 工具：从 FRED（美联储经济数据库）拉取最新宏观指标。

依赖：pip install fredapi；环境变量 FRED_API_KEY
申请地址：https://fred.stlouisfed.org/docs/api/api_key.html

fred_api_key 作为显式参数传入，不在函数内部读环境变量，
调用方（OrchestratorLoop）负责从 config 取得后传入，方便测试。
"""

from typing import Optional

# 指标 key → (FRED 序列 ID, 中文标签)
_FRED_SERIES = {
    "fed_funds_rate": ("FEDFUNDS", "联邦基金利率 (%)"),
    "treasury_10y":   ("DGS10",    "10年期国债收益率 (%)"),
    "cpi_yoy":        ("CPIAUCSL", "CPI 同比变化 (%)"),
    "unemployment":   ("UNRATE",   "失业率 (%)"),
    "vix":            ("VIXCLS",   "VIX 恐慌指数"),
}

# 供 LLM / 报告展示参考的指标说明
_INDICATOR_DOCS = {
    "fed_funds_rate": "联邦基金利率（%），反映货币政策松紧",
    "treasury_10y":   "10年期国债收益率（%），反映长端利率预期",
    "cpi_yoy":        "CPI 同比变化（%），反映通货膨胀水平",
    "unemployment":   "失业率（%），反映劳动力市场状况",
    "vix":            "VIX 恐慌指数，反映市场隐含波动率和风险偏好",
}


def _fetch_fred_indicators(fred_api_key: str) -> dict:
    """
    拉取全部指标的最新值：{key: {"value": float|None, "date": str, "label": str[, "error": str]}}
    单个指标失败时该项 value=None 并附带 error，不影响其他指标。
    CPIAUCSL 是价格指数，换算为同比：最新值 / 13 个点之前的值 - 1。
    """
    # macOS SSL 证书修复（Python 官方安装包不自带系统证书链）。
    # 注意：这会关闭整个进程的 HTTPS 证书校验，更安全的做法是运行
    # Python 自带的 "Install Certificates.command" 后删除这两行。
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("缺少 fredapi 库。请运行：pip install fredapi")

    fred = Fred(api_key=fred_api_key)
    result = {}

    for key, (series_id, label) in _FRED_SERIES.items():
        try:
            series = fred.get_series(series_id).dropna()
            if series.empty:
                result[key] = {"value": None, "date": "N/A", "label": label}
                continue
            latest_val = float(series.iloc[-1])
            if key == "cpi_yoy" and len(series) >= 13:
                latest_val = (latest_val / float(series.iloc[-13]) - 1) * 100
            result[key] = {
                "value": round(latest_val, 2),
                "date":  series.index[-1].strftime("%Y-%m-%d"),
                "label": label,
            }
        except Exception as e:
            result[key] = {"value": None, "date": "N/A", "label": label, "error": str(e)}

    return result


def _format_indicators(indicators: dict) -> str:
    """将指标 dict 格式化为可读文本块"""
    lines = []
    for key, info in indicators.items():
        label = info.get("label", key)
        if info.get("value") is not None:
            lines.append(f"  {label}：{info['value']}  （{info.get('date', 'N/A')}）")
        else:
            lines.append(f"  {label}：N/A  （{info.get('error', '数据不可用')}）")
    return "\n".join(lines)


def query_macro(fred_api_key: Optional[str], indicators: Optional[list[str]] = None) -> dict:
    """
    Parameters
    ----------
    fred_api_key : FRED API Key，未配置时返回 error
    indicators   : 需要查询的指标（fed_funds_rate / treasury_10y / cpi_yoy /
                   unemployment / vix），不传则返回全部

    Returns
    -------
    成功：{"indicators": dict, "summary_text": str, "indicator_docs": dict}
    失败：{"error": str}
    """
    if not fred_api_key:
        return {"error": "未配置 FRED_API_KEY，宏观数据不可用"}

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
