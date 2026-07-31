"""
report_renderer.py
-------------------
把 Orchestrator 的 emit_report 草稿 + Critic 审查结果渲染成 Markdown 报告。

纯函数模块：render_report() 不做任何网络调用/IO，输入输出都是普通
数据结构，方便直接单测（同样的 draft/critique/tool_log 输入，
断言输出的 Markdown 包含哪些段落）。

从 orchestrator.py 的 OrchestratorAgent._render_report 拆分出来。
"""

import json
from datetime import datetime


def render_report(
    entity: str,
    draft: dict,
    critique: dict,
    tool_log: list,
    model_name: str,
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
            f"*置信度调整：{adj_zh}*\n"
        )

    # 工具调用轨迹（可观测性）
    trajectory_lines = []
    for i, entry in enumerate(tool_log, 1):
        trajectory_lines.append(f"{i}. `{entry['tool']}({json.dumps(entry['input'], ensure_ascii=False)})`")
    trajectory_md = "\n".join(trajectory_lines) or "（无工具调用记录）"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# 投资建议报告：{entity}

> 生成时间：{now} | 模型：{model_name} | 配置建议：**{recommendation}** | 置信度：{conf_zh}

---

## 执行摘要

{draft.get('executive_summary', '')}

---

## 宏观环境分析

{draft.get('macro_analysis', '')}

---

## 个股信号分析（FinDKG / WSJ）

{draft.get('entity_analysis', '')}

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

<details>
<summary>Agent 工具调用轨迹（Tool Call Trajectory）</summary>

{trajectory_md}

</details>

---

*本报告由 Multi-Agent 系统自动生成（Orchestrator tool use + Critic reflection），仅供参考，不构成投资建议。*
"""
