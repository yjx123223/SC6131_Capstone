"""
report_renderer.py
-------------------
把 Orchestrator 的 emit_report 草稿 + Critic 审查结果渲染成 Markdown 报告。

纯函数模块：render_report() 不做任何网络调用/IO，输入输出都是普通
数据结构，方便直接单测（同样的 draft/critique/tool_log 输入，
断言输出的 Markdown 包含哪些段落）。

从 orchestrator.py 的 OrchestratorAgent._render_report 拆分出来。

feat/market：正文改为基本面 / 技术面 / 新闻舆情 / 监管申报四块；
新增"数据来源与时效"段落——直接从 tool_log 确定性生成（行情截至日期、
新闻条目与链接、SEC 申报链接、失败的数据源），不经过 LLM，
保证引用可追溯、不会被模型改写或编造。
"""

import json
from datetime import datetime
from typing import Optional

from compliance import disclaimer_block
from live_kg.ontology import EVENT_TYPES, POLARITY_ZH
from tool_log_summary import latest_result


def render_report(
    entity: str,
    draft: dict,
    critique: dict,
    tool_log: list,
    model_name: str,
    compliance: Optional[dict] = None,
) -> str:
    """
    将 emit_report dict 渲染为 Markdown 格式报告。

    Parameters
    ----------
    entity     : 目标实体名
    draft      : OrchestratorLoop 产出的 emit_report 草稿（可能已被 revise 过）
    critique   : CriticAgent.review() 的审查结果
    tool_log   : OrchestratorLoop 的工具调用记录（用于展示调用轨迹）
    model_name : 用于报告头部展示的模型名（Orchestrator 使用的模型）
    compliance : ComplianceChecker.check() 的结果（可选）；传入时展示合规检查
                 段落，并在头部注明置信度是否被规则下调
    """
    recommendation = draft.get("recommendation", "观望")
    confidence     = draft.get("confidence", "low")
    conf_zh        = {"high": "高 ✅", "medium": "中 ⚠️", "low": "低 ❗"}.get(confidence, confidence)

    # 关键信号列表
    signals_md = "\n".join(
        f"- {s}" for s in draft.get("key_signals", [])
    ) or "- （无）"

    # Critic 审查备注
    conflicts = critique.get("conflicts", [])
    critic_section = ""
    if conflicts:
        conflicts_md = "\n".join(f"- {c}" for c in conflicts)
        adj = critique.get("confidence_adjustment", "maintain")
        adj_zh = {"lower": "已下调", "raise": "已上调", "maintain": "维持"}.get(adj, adj)
        critic_section = (
            f"\n## ⚠️ Critic Agent 审查备注\n"
            f"{conflicts_md}\n\n"
            f"*置信度调整：{adj_zh}*\n\n"   # 末尾空行：避免紧跟的 --- 把这一行变成 setext 标题
        )

    sources_md = render_data_sources(tool_log)
    graph_md = render_knowledge_graph(latest_result(tool_log, "query_company_graph"),
                                      draft.get("cited_event_ids", []))
    compliance_md = render_compliance(compliance) if compliance else ""

    conf_note = ""
    if compliance:
        c = compliance.get("confidence", {})
        if c.get("original") and c.get("final") and c["original"] != c["final"]:
            orig_zh = {"high": "高", "medium": "中", "low": "低"}.get(c["original"], c["original"])
            conf_note = f"（原为{orig_zh}，已按审查/合规规则下调）"
    sentiment_zh = {
        "positive": "偏正面", "neutral": "中性", "negative": "偏负面", "unavailable": "数据不可用",
    }.get(draft.get("news_sentiment", ""), draft.get("news_sentiment", "") or "未标注")

    # 工具调用轨迹（可观测性）
    trajectory_lines = []
    for i, entry in enumerate(tool_log, 1):
        trajectory_lines.append(f"{i}. `{entry['tool']}({json.dumps(entry['input'], ensure_ascii=False)})`")
    trajectory_md = "\n".join(trajectory_lines) or "（无工具调用记录）"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# 投资建议报告：{entity}

> 生成时间：{now} | 模型：{model_name} | 配置建议：**{recommendation}** | 置信度：{conf_zh}{conf_note}

---

## 执行摘要

{draft.get('executive_summary', '')}

---

## 宏观环境分析

{draft.get('macro_analysis', '')}

---

## 基本面与估值

{draft.get('fundamental_analysis', '')}

---

## 技术面

{draft.get('technical_analysis', '')}

---

## 新闻舆情（整体情绪：{sentiment_zh}）

{draft.get('news_sentiment_analysis', '')}

---

## 监管申报（SEC EDGAR）

{draft.get('filings_analysis', '')}

---

## 产业链与竞争格局（知识图谱）

{draft.get('supply_chain_analysis', '')}

{graph_md}
---

## 综合配置建议

**{recommendation}**

{draft.get('recommendation_rationale', '')}

---

## 关键信号

{signals_md}

---

## 风险提示

{draft.get('risk_warnings', '')}
{critic_section}---

## 数据来源与时效

{sources_md}

---
{compliance_md}
<details>
<summary>Agent 工具调用轨迹（Tool Call Trajectory）</summary>

{trajectory_md}

</details>

---

{disclaimer_block(now)}
*本报告由 Multi-Agent 系统自动生成（Orchestrator tool use + 实时知识图谱 + Critic reflection + 规则合规检查），仅供参考，不构成投资建议。*
"""


def render_data_sources(tool_log: list) -> str:
    """根据工具调用结果生成数据来源段落（纯函数，不经过 LLM）"""
    lines = []
    for entry in tool_log:
        tool = entry.get("tool", "")
        r = entry.get("result") or {}
        if "error" in r:
            lines.append(f"- ❌ `{tool}` 数据不可用：{r['error']}")
            continue
        if tool == "query_market_data":
            lines.append(
                f"- 行情与财务指标：{r.get('source')}，{r.get('ticker')} 行情截至 **{r.get('data_as_of')}**"
                f"（回看 {r.get('period')}）"
            )
            for w in r.get("warnings", []):
                lines.append(f"  - ⚠️ {w}")
        elif tool == "query_news":
            lines.append(f"- 新闻：{r.get('source')}，近 {r.get('lookback_days')} 天 {r.get('article_count', 0)} 条")
            for a in r.get("articles", []):
                title = a.get("title", "")
                link = f"[{title}]({a['url']})" if a.get("url") else title
                lines.append(f"  - {a.get('published_at', '')[:10]} · {a.get('publisher', '')} · {link}")
        elif tool == "query_sec_filings":
            lines.append(f"- 监管申报：{r.get('source')}，{r.get('company')}（CIK {r.get('cik')}）")
            for f in r.get("filings", []):
                name = f"{f.get('form')} {f.get('filing_date')}"
                link = f"[{name}]({f['url']})" if f.get("url") else name
                lines.append(f"  - {link}")
        elif tool == "query_company_graph":
            st = r.get("stats", {})
            ext = st.get("extraction", {})
            lines.append(
                f"- 知识图谱：{r.get('source')}，{st.get('node_count', 0)} 个节点 / {st.get('edge_count', 0)} 条边，"
                f"抽取事件 {ext.get('events_extracted', 0)} 个"
            )
            for w in r.get("warnings", []):
                lines.append(f"  - ⚠️ {w}")
        elif tool == "query_macro":
            dates = sorted({v.get("date") for v in (r.get("indicators") or {}).values()
                            if isinstance(v, dict) and v.get("date") not in (None, "N/A")})
            span = f"，指标日期 {dates[0]} ~ {dates[-1]}" if dates else ""
            lines.append(f"- 宏观指标：FRED（美联储经济数据库）{span}")
    return "\n".join(lines) or "（本次未调用任何数据工具）"


_SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}


def render_compliance(compliance: dict) -> str:
    """合规检查段落（纯函数）"""
    status = "✅ 通过" if compliance.get("is_compliant") else "⚠️ 未通过"
    lines = [
        "",
        "## 合规检查",
        "",
        f"**状态**：{status} | **评分**：{compliance.get('score', 0)}/100",
        "",
    ]
    issues = compliance.get("issues", [])
    if issues:
        lines += ["| 级别 | 类别 | 位置 | 说明 |", "|---|---|---|---|"]
        for i in issues:
            desc = str(i.get("description", "")).replace("|", "\\|")
            lines.append(
                f"| {_SEVERITY_ICON.get(i.get('severity'), '')} {i.get('severity')} "
                f"| {i.get('category')} | {i.get('field') or '-'} | {desc} |"
            )
    else:
        lines.append("未发现问题。")
    lines += ["", "---", ""]
    return "\n".join(lines)


def _md_cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def _link(title: str, url: str) -> str:
    title = _md_cell(title)
    return f"[{title}]({url})" if url else title


def render_knowledge_graph(graph: Optional[dict], cited_event_ids=()) -> str:
    """
    知识图谱详情（纯函数）：Mermaid 关系图 + 风险传导表 + 潜在利好 + 目标公司事件 + 构建统计。
    数据全部来自 query_company_graph 的结果（含不发给模型的 "_" 字段），不经过 LLM。
    报告中引用过的事件编号用 ★ 标记。
    """
    if not graph:
        return "*知识图谱不可用（本次未成功构建）。*\n"

    cited = {str(x).upper() for x in (cited_event_ids or [])}
    star = lambda eid: f"★{eid}" if eid in cited else eid
    type_zh = lambda t: EVENT_TYPES.get(t, t)

    parts = ["<details open>", "<summary>知识图谱详情（点击折叠）</summary>", ""]

    if graph.get("_mermaid"):
        parts += ["```mermaid", graph["_mermaid"], "```", ""]

    neighbors = graph.get("neighbors", [])
    if neighbors:
        parts += ["**相关公司**", "", "| 公司 | 跳数 | 与目标的关系 | 关系路径 |", "|---|---|---|---|"]
        for n in neighbors:
            parts.append(f"| {n['ticker']} | {n['hop']} | {'、'.join(n['roles'])} | {_md_cell(n['path'])} |")
        parts.append("")

    risks = graph.get("_risks", [])
    parts += ["**风险传导（规则推理）**", ""]
    if risks:
        parts += ["| 事件 | 影响 | 分数 | 来源公司 | 日期 | 事件内容 | 传导路径 | 新闻 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in risks:
            src = r["sources"][0] if r.get("sources") else {}
            parts.append(
                f"| {star(r['event_id'])} | {r['impact']} | {r['score']:.2f} | {r['neighbor']}（{r['role_zh']}） "
                f"| {r['date']} | {type_zh(r['event_type'])}：{_md_cell(r['summary'])} | {_md_cell(r['path'])} "
                f"| {_link(src.get('title', ''), src.get('url', ''))} |"
            )
    else:
        parts.append("未发现相关公司的风险传导事件。")
    parts.append("")

    opps = graph.get("_opportunities", [])
    if opps:
        parts += ["**潜在利好（竞争对手承压）**", ""]
        for o in opps:
            parts.append(f"- {star(o['event_id'])} {o['neighbor']}（{o['date']}）：{_md_cell(o['summary'])}")
        parts.append("")

    events = graph.get("_target_events", [])
    if events:
        parts += ["**目标公司自身事件**", "", "| 事件 | 日期 | 类型 | 方向 | 内容 | 新闻 |", "|---|---|---|---|---|---|"]
        for e in events:
            src = e["sources"][0] if e.get("sources") else {}
            parts.append(
                f"| {star(e['id'])} | {e['date']} | {type_zh(e['event_type'])} | {POLARITY_ZH.get(e['polarity'], e['polarity'])} "
                f"| {_md_cell(e['summary'])} | {_link(src.get('title', ''), src.get('url', ''))} |"
            )
        parts.append("")

    st = graph.get("stats", {})
    ext = st.get("extraction", {})
    dropped = "、".join(f"{k} {v}" for k, v in ext.get("dropped", {}).items()) or "无"
    parts += [
        f"*构建统计：节点 {st.get('node_count', 0)}，边 {st.get('edge_count', 0)}；"
        f"事件抽取成功 {len(ext.get('companies_ok', []))} 家、失败 {len(ext.get('companies_failed', []))} 家；"
        f"抽取结果被过滤：{dropped}；本体校验拒绝 {st.get('rejected_count', 0)} 条。*",
        "",
        "*说明：公司间的供应/竞争/合作关系来自人工整理的种子数据；事件由 LLM 从近期新闻中抽取；"
        "风险传导为基于规则的推断，分数 = 关系权重 × 跳数衰减 × 事件置信度，仅供参考。*",
        "",
        "</details>",
        "",
    ]
    return "\n".join(parts)
