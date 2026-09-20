"""
tool_log_summary.py
--------------------
把 OrchestratorLoop 的工具调用记录（tool_log）压缩成文字摘要，
供 Critic 审查和 revise 修订时作为"原始数据依据"。

Critic 和 revise() 共用同一份摘要（让修订只能基于真实数据，而不是凭空改写）。
工具返回 error 时也要写进摘要——Critic 需要知道哪些数据缺失，
才能判断报告有没有"无数据却下结论"。

重要约束：摘要必须覆盖报告可能引用的**全部事实字段**。摘要漏掉的字段，
Critic 会当成"编造"而要求删除，revise 又只能看到同一份摘要，于是真实有据的
内容被删掉。所以行情的技术面/基本面字段、收盘序列、新闻的正文摘要都要完整带上，
不做条数截断，未知字段也照原样输出——省下的 token 远不值一次误删。
"""


def _fmt_kv(d: dict, labels: dict) -> str:
    """字典逐项展开；None 跳过。未知键用原始键名，保证新增字段不会被悄悄吞掉。"""
    return "，".join(f"{labels.get(k, k)} {v}" for k, v in d.items() if v is not None)


def _fmt_market(r: dict) -> str:
    """行情摘要。技术面与基本面**全量**展开——报告会引用哪个字段无法预知，
    漏一个，Critic 就会把有据可查的数字判成编造（实测 return_5d_pct /
    return_period_pct / ROE 都栽在这上面）。"""
    head = (f"行情[{r.get('ticker')}] 截至 {r.get('data_as_of')}"
            f"（区间 {r.get('period')}，来源 {r.get('source')}）")
    lines = [head]
    if r.get("technicals"):
        lines.append("  技术面：" + _fmt_kv(r["technicals"], _TECH_LABELS))
    if r.get("fundamentals"):
        lines.append("  基本面：" + _fmt_kv(r["fundamentals"], _FUND_LABELS))
    closes = r.get("recent_closes") or []
    if closes:
        series = "；".join(f"{c.get('date')} {c.get('close')}" for c in closes)
        lines.append(f"  近{len(closes)}个交易日收盘：{series}")
    if r.get("warnings"):
        lines.append(f"  警告：{'; '.join(r['warnings'])}")
    return "\n".join(lines)


_TECH_LABELS = {
    "last_close": "收盘", "ma20": "MA20", "ma50": "MA50",
    "price_vs_ma20": "相对MA20", "price_vs_ma50": "相对MA50",
    "rsi14": "RSI14", "rsi14_signal": "RSI信号",
    "volatility_20d_pct": "20日日波动率%", "volatility_20d_annualized_pct": "20日年化波动率%",
    "return_5d_pct": "5日涨跌%", "return_20d_pct": "20日涨跌%", "return_period_pct": "区间涨跌%",
}

_FUND_LABELS = {
    "market_cap": "市值", "trailing_pe": "PE(TTM)", "forward_pe": "预期PE",
    "price_to_book": "PB", "dividend_yield_pct": "股息率%",
    "profit_margin_pct": "净利率%", "revenue_growth_pct": "营收增速%",
    "earnings_growth_pct": "净利增速%", "return_on_equity_pct": "ROE%",
    "debt_to_equity": "负债权益比", "current_ratio": "流动比率",
    "total_revenue": "营业收入", "net_income": "净利润", "free_cash_flow": "自由现金流",
    "beta": "beta", "fifty_two_week_high": "52周高", "fifty_two_week_low": "52周低",
    "analyst_target_mean": "分析师目标价", "analyst_recommendation": "分析师评级",
    "analyst_count": "分析师数量",
}


def _fmt_news(r: dict) -> str:
    arts = r.get("articles", [])
    head = f"新闻[{r.get('ticker')}]：近{r.get('lookback_days')}天 {len(arts)} 条"
    lines = []
    for a in arts:                      # 不截断条数：被截掉的那条正是报告可能引用的
        lines.append(f"  · {a.get('published_at', '')[:10]} {a.get('publisher')}: {a.get('title')}")
        if a.get("summary"):            # 正文摘要是报告论据的主要来源，必须带上
            lines.append(f"    摘要：{a['summary']}")
    return "\n".join([head] + lines)


def _fmt_sec(r: dict) -> str:
    items = []
    for x in r.get("filings", []):
        dates = f"{x.get('filing_date')}"
        if x.get("report_date"):
            dates += f"，报告期 {x['report_date']}"
        items.append(f"{x.get('form')}({dates})")
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
