"""
tool_log_summary.py
--------------------
把 OrchestratorLoop 的工具调用记录（tool_log）压缩成文字摘要，
供 Critic 审查和 revise 修订时作为"原始数据依据"。

Critic 和 revise() 共用同一份摘要（让修订只能基于真实数据，而不是凭空改写）。
工具返回 error 时也要写进摘要——Critic 需要知道哪些数据缺失，
才能判断报告有没有"无数据却下结论"。
"""


def _fmt_market(r: dict) -> str:
    t = r.get("technicals", {})
    f = r.get("fundamentals", {})
    parts = [
        f"行情[{r.get('ticker')}] 截至 {r.get('data_as_of')}：收盘 {t.get('last_close')}",
        f"MA20 {t.get('ma20')}（{t.get('price_vs_ma20')}）",
        f"MA50 {t.get('ma50')}（{t.get('price_vs_ma50')}）",
        f"RSI14 {t.get('rsi14')}（{t.get('rsi14_signal')}）",
        f"20日涨跌 {t.get('return_20d_pct')}%",
        f"年化波动率 {t.get('volatility_20d_annualized_pct')}%",
        f"PE {f.get('trailing_pe')}",
        f"利润率 {f.get('profit_margin_pct')}%",
        f"营收增速 {f.get('revenue_growth_pct')}%",
        f"分析师目标价 {f.get('analyst_target_mean')}（{f.get('analyst_recommendation')}）",
    ]
    text = "，".join(parts)
    if r.get("warnings"):
        text += f"；警告：{'; '.join(r['warnings'])}"
    return text


def _fmt_news(r: dict) -> str:
    arts = r.get("articles", [])
    head = f"新闻[{r.get('ticker')}]：近{r.get('lookback_days')}天 {len(arts)} 条"
    titles = [f"  · {a.get('published_at', '')[:10]} {a.get('publisher')}: {a.get('title')}" for a in arts[:6]]
    return "\n".join([head] + titles)


def _fmt_sec(r: dict) -> str:
    items = [f"{x.get('form')}({x.get('filing_date')})" for x in r.get("filings", [])]
    return f"SEC 申报[{r.get('ticker')}]：" + ("、".join(items) or "无")


def _fmt_graph(r: dict) -> str:
    stats = r.get("stats", {})
    lines = [
        f"知识图谱[{r.get('ticker')}]：{stats.get('node_count', 0)} 个节点、{stats.get('edge_count', 0)} 条边；"
        f"邻居 " + (", ".join(
            f"{n['ticker']}({'/'.join(n.get('roles', []))}{', 经 ' + n['via'] if n.get('via') else ''})"
            for n in r.get("neighbors", [])
        ) or "无"),
    ]
    for e in r.get("target_events", []):
        lines.append(f"  · 目标事件 {e['id']} {e['date']} {e['polarity']} {e['event_type']}：{e['summary']}")
    for x in r.get("propagated_risks", []):
        lines.append(
            f"  · 传导风险 {x['event_id']} {x['impact']}（分数 {x['score']}）{x['date']} "
            f"{x['neighbor']}：{x['summary']}｜路径 {x['path']}"
        )
    for x in r.get("opportunities", []):
        lines.append(f"  · 潜在利好 {x['event_id']} {x['impact']} {x['neighbor']}：{x['summary']}")
    if r.get("warnings"):
        lines.append(f"  · 图谱警告：{'; '.join(r['warnings'])}")
    return "\n".join(lines)


_FORMATTERS = {
    "query_market_data":  _fmt_market,
    "query_news":         _fmt_news,
    "query_sec_filings":  _fmt_sec,
    "query_macro":        lambda r: f"宏观指标：\n{r.get('summary_text', '无数据')}",
    "query_company_graph": _fmt_graph,
}


def latest_result(tool_log: list, tool: str) -> dict | None:
    """某个工具最后一次成功的结果；没有则返回 None"""
    for entry in reversed(tool_log):
        result = entry.get("result") or {}
        if entry.get("tool") == tool and "error" not in result:
            return result
    return None


def summarize_tool_log(tool_log: list) -> str:
    parts = []
    for entry in tool_log:
        tool = entry.get("tool", "")
        result = entry.get("result") or {}
        if "error" in result:
            parts.append(f"{tool}：❌ 数据不可用（{result['error']}）")
            continue
        fmt = _FORMATTERS.get(tool)
        parts.append(fmt(result) if fmt else f"{tool}：已调用")
    return "\n".join(parts) if parts else "（无工具调用记录）"
