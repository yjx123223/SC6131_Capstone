"""
orchestrator.py
---------------
Orchestrator Agent（改法 A：Tool Use 架构）

架构说明：
  Orchestrator 是一个真正的 Agent——它持有4个工具定义，
  由 Claude 自主决定调用哪些工具、调用顺序和参数，
  直到调用 emit_report 输出结构化草稿。

  草稿完成后，Critic Agent 独立审查信号冲突与过度自信，
  若发现问题，Orchestrator 进行一轮修订后输出最终报告。

工具列表：
  - query_kg_signals(entity, weeks)     → FinDKG KG 历史事件 + 预测信号
  - query_macro(indicators)             → FRED 宏观指标
  - get_feedback_stats(relation_type)   → 历史建议评分统计
  - emit_report(schema)                 → 输出结构化报告草稿（结束 loop）
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic


# ── Tool 定义 ────────────────────────────────────────────────────

_TOOL_DEFINITIONS = [
    {
        "name": "query_kg_signals",
        "description": (
            "查询目标实体在 FinDKG 知识图谱中的历史事件信号，"
            "包括正面影响事件（Positive_Impact_On / Raise / Invests_In）、"
            "负面影响事件（Negative_Impact_On / Decrease）、其他关联事件，"
            "以及 KGTransformer 的下一周预测信号（若可用）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "实体名称，如 'Apple Inc.'",
                },
                "weeks": {
                    "type": "integer",
                    "description": "查询最近 N 周，默认 12",
                    "default": 12,
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "query_macro",
        "description": (
            "从 FRED（美联储经济数据库）获取最新宏观经济指标，"
            "可选指标：fed_funds_rate（联邦基金利率）、treasury_10y（10年期国债收益率）、"
            "cpi_yoy（CPI同比）、unemployment（失业率）、vix（VIX恐慌指数）。"
            "不传 indicators 则返回全部指标。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "需要查询的指标列表，可选值："
                        "fed_funds_rate, treasury_10y, cpi_yoy, unemployment, vix"
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_feedback_stats",
        "description": (
            "获取历史建议的用户评分统计，了解哪类 KG 信号关系类型"
            "在过去的建议中表现更好（平均评分更高）。"
            "可按具体关系类型过滤，或不传参数获取全部统计。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relation_type": {
                    "type": "string",
                    "description": (
                        "关系类型，如 'Positive_Impact_On'，"
                        "留空则返回所有关系类型的统计"
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "emit_report",
        "description": (
            "当你已收集足够信息，调用此工具输出结构化投资建议报告草稿。"
            "调用后 Orchestrator 循环结束，草稿将进入 Critic Agent 审查。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "执行摘要（2-3句话，核心结论）",
                },
                "macro_analysis": {
                    "type": "string",
                    "description": "宏观环境分析",
                },
                "entity_analysis": {
                    "type": "string",
                    "description": "目标实体的KG信号分析",
                },
                "recommendation": {
                    "type": "string",
                    "enum": ["增持", "持有", "观望", "减仓", "回避"],
                    "description": "配置建议",
                },
                "recommendation_rationale": {
                    "type": "string",
                    "description": "配置建议的具体理由",
                },
                "risk_warnings": {
                    "type": "string",
                    "description": "主要风险提示",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "整体置信度",
                },
                "key_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "支撑结论的关键信号列表（3-5条）",
                },
            },
            "required": [
                "executive_summary",
                "macro_analysis",
                "entity_analysis",
                "recommendation",
                "recommendation_rationale",
                "risk_warnings",
                "confidence",
                "key_signals",
            ],
        },
    },
]


class OrchestratorAgent:
    """
    Tool Use 架构的 Orchestrator Agent。

    使用示例
    --------
    >>> from kg_query import FinDKGGraph
    >>> from kg_predictor import KGPredictor
    >>> from feedback_store import FeedbackStore
    >>> from orchestrator import OrchestratorAgent
    >>>
    >>> graph     = FinDKGGraph()
    >>> predictor = KGPredictor()
    >>> store     = FeedbackStore()
    >>> orch      = OrchestratorAgent()
    >>>
    >>> report = orch.generate_report("Apple Inc.", graph,
    ...                               predictor=predictor,
    ...                               feedback_store=store)
    >>> print(report)
    """

    MAX_ITERATIONS = 10   # agentic loop 最大轮次（防止无限循环）

    def __init__(
        self,
        api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 2048,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "未找到 Anthropic API Key。\n"
                "请设置：export ANTHROPIC_API_KEY='your-key'"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")

    # ── 主入口 ──────────────────────────────────────────────────────

    def generate_report(
        self,
        entity: str,
        graph,
        predictor=None,
        feedback_store=None,
        weeks: int = 12,
    ) -> str:
        """
        完整 Multi-Agent 流程：
          Orchestrator loop (tool use) → Critic 审查 → (可选)修订 → 渲染报告 → 保存

        Parameters
        ----------
        entity         : 目标实体名
        graph          : FinDKGGraph 实例
        predictor      : KGPredictor 实例（可选）
        feedback_store : FeedbackStore 实例（可选）
        weeks          : 查询最近多少周

        Returns
        -------
        Markdown 格式的最终投资建议报告
        """
        context = {
            "graph": graph,
            "predictor": predictor,
            "feedback_store": feedback_store,
            "default_weeks": weeks,
        }

        # 1. Orchestrator agentic loop
        print(f"\n[Orchestrator] 启动 Agent Loop — 目标实体：{entity}")
        draft, tool_log = self._run_orchestrator_loop(entity, weeks, context)

        if draft is None:
            return f"[Orchestrator] ⚠️ Agent 未能生成报告草稿（超出最大迭代次数或异常退出）"

        # 2. Critic Agent 审查
        print(f"\n[Critic] 审查草稿报告...")
        critique = self._run_critic(entity, draft, tool_log)
        approved = critique.get("approved", True)
        conflicts = critique.get("conflicts", [])

        if conflicts:
            print(f"[Critic] 发现 {len(conflicts)} 个问题：" + "；".join(conflicts[:2]))
        else:
            print(f"[Critic] 审查通过，无信号冲突")

        # 3. 若 Critic 不通过，进行一轮修订
        if not approved and critique.get("suggestions"):
            print(f"[Orchestrator] 根据 Critic 建议修订报告...")
            draft = self._revise_draft(entity, draft, critique, tool_log)

        # 4. 渲染为 Markdown
        report_md = self._render_report(entity, draft, critique, tool_log)

        # 5. 保存到本地
        self._save_report(entity, report_md)

        return report_md

    # ── Orchestrator Loop ────────────────────────────────────────────

    def _run_orchestrator_loop(
        self, entity: str, weeks: int, context: dict
    ) -> tuple[Optional[dict], list]:
        """
        核心 agentic loop：Claude 自主决定调用哪些工具、顺序和参数，
        直到调用 emit_report 为止。

        Returns
        -------
        (draft_dict, tool_call_log)
        """
        system_prompt = """你是一位专业的金融研究员 Agent。
你的任务是为指定实体生成投资建议报告。

你有以下工具可以使用：
- query_kg_signals：查询该实体在金融知识图谱中的历史事件和预测信号
- query_macro：获取当前宏观经济指标（利率、通胀、VIX等）
- get_feedback_stats：查看历史建议的评分统计，了解哪类信号更可信

工作流程建议（你可以根据情况调整）：
1. 先查询 KG 信号，了解实体的基本面事件
2. 再查询宏观指标，判断整体市场环境
3. 可选：查询历史评分，调整信号权重
4. 当信息足够时，调用 emit_report 输出结构化报告

注意：
- 宏观信号决定整体仓位方向，KG信号决定个股判断
- 若两类信号方向相反，在报告中明确标注冲突并说明如何取舍
- emit_report 的 confidence 需真实反映数据质量，不要过度自信"""

        initial_message = (
            f"请为 [{entity}] 生成投资建议报告，查询最近 {weeks} 周的 KG 数据。"
        )

        messages = [{"role": "user", "content": initial_message}]
        tool_log = []
        draft = None

        for iteration in range(self.MAX_ITERATIONS):
            print(f"[Orchestrator] 第 {iteration + 1} 轮推理...")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=_TOOL_DEFINITIONS,
                messages=messages,
            )

            # 将 assistant 响应加入历史
            messages.append({"role": "assistant", "content": response.content})

            # 无工具调用，Claude 直接结束
            if response.stop_reason == "end_turn":
                print("[Orchestrator] Agent 自然结束（未调用 emit_report）")
                break

            if response.stop_reason != "tool_use":
                break

            # 处理所有 tool_use 块
            tool_results = []
            emit_called = False

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input
                tool_id    = block.id

                if tool_name == "emit_report":
                    draft = tool_input
                    emit_called = True
                    print(f"[Orchestrator] emit_report 调用 → 草稿已捕获")
                    tool_results.append({
                        "type": "tool_use_id",   # 占位，下面统一构建
                        "_id": tool_id,
                        "_content": '{"status": "草稿已接收，进入 Critic 审查阶段"}',
                    })
                else:
                    print(f"[Orchestrator] 工具调用：{tool_name}({tool_input})")
                    result = self._execute_tool(tool_name, tool_input, context)
                    tool_log.append({
                        "tool":   tool_name,
                        "input":  tool_input,
                        "result": result,
                    })
                    tool_results.append({
                        "type": "tool_use_id",
                        "_id": tool_id,
                        "_content": json.dumps(result, ensure_ascii=False, default=str),
                    })

            # 构建标准 tool_result 消息
            content_blocks = []
            for tr in tool_results:
                content_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": tr["_id"],
                    "content":     tr["_content"],
                })
            messages.append({"role": "user", "content": content_blocks})

            if emit_called:
                break

        return draft, tool_log

    # ── 工具执行器 ───────────────────────────────────────────────────

    def _execute_tool(self, name: str, tool_input: dict, context: dict) -> dict:
        """根据工具名分发执行"""
        if name == "query_kg_signals":
            result = self._tool_query_kg(tool_input, context)
        elif name == "query_macro":
            result = self._tool_query_macro(tool_input)
        elif name == "get_feedback_stats":
            result = self._tool_feedback_stats(tool_input, context)
        else:
            result = {"error": f"未知工具：{name}"}

        if "error" in result:
            print(f"[Orchestrator] ⚠️  {name} 错误：{result['error']}")

        return result

    def _tool_query_kg(self, tool_input: dict, context: dict) -> dict:
        """query_kg_signals 工具实现"""
        graph     = context["graph"]
        predictor = context.get("predictor")
        entity    = tool_input.get("entity", "")
        weeks     = tool_input.get("weeks", context.get("default_weeks", 12))

        summary = graph.get_impact_summary(entity, n_recent_weeks=weeks)

        if summary.get("total_events", 0) == 0:
            return {"error": f"KG 中未找到 '{entity}' 的近期数据"}

        predictions = None
        if predictor and predictor.available:
            predictions = predictor.predict_next(entity, graph, top_k=8)

        def fmt(events):
            return [
                {
                    "subject":  e["subject"],
                    "relation": e["relation"],
                    "object":   e["object"],
                    "count":    e["count"],
                }
                for e in events[:6]
            ]

        return {
            "entity":           entity,
            "period":           summary.get("period", "未知"),
            "total_events":     summary.get("total_events", 0),
            "positive_impacts": fmt(summary.get("positive_impacts", [])),
            "negative_impacts": fmt(summary.get("negative_impacts", [])),
            "other_relations":  fmt(summary.get("other_relations", [])),
            "predictions":      predictions or [],
            "has_predictions":  predictions is not None,
        }

    def _tool_query_macro(self, tool_input: dict) -> dict:
        """query_macro 工具实现"""
        if not self.fred_api_key:
            return {"error": "未配置 FRED_API_KEY，宏观数据不可用"}

        try:
            from macro_agent import _fetch_fred_indicators, _format_indicators
        except ImportError:
            return {"error": "macro_agent 模块不可用"}

        requested = set(tool_input.get("indicators") or [])

        try:
            all_indicators = _fetch_fred_indicators(self.fred_api_key)
        except Exception as e:
            return {"error": f"FRED 数据拉取失败：{e}"}

        if requested:
            indicators = {k: v for k, v in all_indicators.items() if k in requested}
        else:
            indicators = all_indicators

        return {
            "indicators":   indicators,
            "summary_text": _format_indicators(indicators),
        }

    def _tool_feedback_stats(self, tool_input: dict, context: dict) -> dict:
        """get_feedback_stats 工具实现"""
        store = context.get("feedback_store")
        if store is None:
            return {"error": "未配置 FeedbackStore，历史评分不可用"}

        try:
            report = store.signal_accuracy_report()
        except Exception as e:
            return {"error": f"评分查询失败：{e}"}

        rel_filter = tool_input.get("relation_type")
        stats = report.get("signal_stats", {})

        if rel_filter:
            stats = {k: v for k, v in stats.items() if rel_filter.lower() in k.lower()}

        return {
            "total_rated":   report.get("total_rated", 0),
            "positive_rate": report.get("positive_rate", 0),
            "signal_stats":  stats,
        }

    # ── Critic Agent ─────────────────────────────────────────────────

    def _run_critic(self, entity: str, draft: dict, tool_log: list) -> dict:
        """
        Critic Agent：独立审查草稿，检查信号冲突和过度自信。

        Returns
        -------
        {
            "approved": bool,
            "conflicts": list[str],
            "confidence_adjustment": "maintain"|"lower"|"raise",
            "suggestions": str,
        }
        """
        # 构建工具调用上下文摘要
        context_parts = []
        for entry in tool_log:
            tool = entry["tool"]
            result = entry["result"]
            if tool == "query_kg_signals":
                pos = len(result.get("positive_impacts", []))
                neg = len(result.get("negative_impacts", []))
                context_parts.append(
                    f"KG信号 [{result.get('entity')}]：总事件{result.get('total_events', 0)}条，"
                    f"正面{pos}类，负面{neg}类"
                )
            elif tool == "query_macro":
                context_parts.append(
                    f"宏观指标：{result.get('summary_text', '无数据')}"
                )
            elif tool == "get_feedback_stats":
                context_parts.append(
                    f"历史评分：已评{result.get('total_rated', 0)}次，"
                    f"正面率{result.get('positive_rate', 0):.1%}"
                )

        context_summary = "\n".join(context_parts) if context_parts else "（无工具调用记录）"

        system_prompt = """你是一位严格的风险审查员（Critic Agent）。
你将收到一份投资建议报告草稿和生成该报告所用的原始数据摘要。

检查以下三点：
1. 信号冲突：KG信号方向与宏观信号方向是否矛盾？若有冲突，报告是否正确处理？
2. 过度自信：置信度（high/medium/low）是否与数据质量匹配？数据稀少时不应标为 high。
3. 引用准确性：报告结论是否有对应的原始数据支撑？是否存在无依据的推断？

输出严格的 JSON 格式，不要输出其他内容：
{
  "approved": true 或 false,
  "conflicts": ["问题描述1", "问题描述2"],
  "confidence_adjustment": "maintain" 或 "lower" 或 "raise",
  "suggestions": "给 Orchestrator 的具体修改建议（approved=true 时为空字符串）"
}"""

        user_prompt = f"""原始数据摘要：
{context_summary}

报告草稿（{entity}）：
- 执行摘要：{draft.get('executive_summary', '')}
- 宏观分析：{draft.get('macro_analysis', '')}
- 个股分析：{draft.get('entity_analysis', '')}
- 配置建议：{draft.get('recommendation', '')}（置信度：{draft.get('confidence', '')}）
- 建议理由：{draft.get('recommendation_rationale', '')}
- 风险提示：{draft.get('risk_warnings', '')}
- 关键信号：{', '.join(draft.get('key_signals', []))}

请输出 JSON 格式的审查结果。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end <= 0:
                print(f"[Critic] 响应中未找到 JSON，原始内容：{raw[:100]!r}")
                raise ValueError("no JSON in response")
            return json.loads(raw[start:end])
        except Exception as e:
            print(f"[Critic] 解析失败，默认通过：{e}")
            return {
                "approved": True,
                "conflicts": [],
                "confidence_adjustment": "maintain",
                "suggestions": "",
            }

    # ── 修订 ─────────────────────────────────────────────────────────

    def _revise_draft(
        self, entity: str, draft: dict, critique: dict, tool_log: list
    ) -> dict:
        """
        Critic 不通过时进行修订。
        使用 tool_use 强制调用 emit_report，避免 JSON 转义问题。
        """
        conflicts_text = "\n".join(f"- {c}" for c in critique.get("conflicts", []))
        suggestions    = critique.get("suggestions", "")

        emit_tool = next(t for t in _TOOL_DEFINITIONS if t["name"] == "emit_report")

        system_prompt = "你是一位专业的金融研究员，正在修订一份投资建议报告。修订完成后必须调用 emit_report 输出结果。"
        user_prompt = f"""以下报告草稿被 Critic Agent 标记为需要修订：

【当前草稿】
执行摘要：{draft.get('executive_summary', '')}
宏观分析：{draft.get('macro_analysis', '')}
个股分析：{draft.get('entity_analysis', '')}
配置建议：{draft.get('recommendation', '')}（置信度：{draft.get('confidence', '')}）
建议理由：{draft.get('recommendation_rationale', '')}
风险提示：{draft.get('risk_warnings', '')}
关键信号：{', '.join(draft.get('key_signals', []))}

【Critic 发现的问题】
{conflicts_text}

【修改建议】
{suggestions}

请根据以上问题修订报告，然后调用 emit_report 输出修订后的完整版本。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=[emit_tool],
                tool_choice={"type": "tool", "name": "emit_report"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "emit_report":
                    print("[Orchestrator] 修订完成")
                    return block.input
            print("[Orchestrator] 修订未找到 emit_report，使用原草稿")
            return draft
        except Exception as e:
            print(f"[Orchestrator] 修订失败，使用原草稿：{e}")
            return draft

    # ── 渲染 & 保存 ──────────────────────────────────────────────────

    def _render_report(
        self,
        entity: str,
        draft: dict,
        critique: dict,
        tool_log: list,
    ) -> str:
        """将 emit_report dict 渲染为 Markdown 格式报告"""
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

> 生成时间：{now} | 模型：{self.model} | 配置建议：**{recommendation}** | 置信度：{conf_zh}

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
{critic_section}
---

<details>
<summary>Agent 工具调用轨迹（Tool Call Trajectory）</summary>

{trajectory_md}

</details>

---

*本报告由 Multi-Agent 系统自动生成（Orchestrator tool use + Critic reflection），仅供参考，不构成投资建议。*
"""

    def _save_report(self, entity: str, report_text: str) -> Path:
        """保存报告为 Markdown 文件"""
        try:
            from config import REPORTS_DIR
            reports_dir = Path(REPORTS_DIR)
        except ImportError:
            reports_dir = Path(__file__).parent / "reports"

        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r"[^\w\-]", "_", entity)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath  = reports_dir / f"{safe_name}_{timestamp}.md"
        filepath.write_text(report_text, encoding="utf-8")
        print(f"[Orchestrator] 报告已保存：{filepath}")
        return filepath

    # ── 多实体对比（保持兼容）──────────────────────────────────────

    def generate_comparison_report(
        self,
        entities: list[str],
        graph,
        predictor=None,
        feedback_store=None,
        weeks: int = 12,
    ) -> str:
        """
        多实体对比：分别为每个实体运行完整 Agent 流程，最后合并对比摘要。
        """
        reports = {}
        for entity in entities:
            print(f"\n{'='*50}\n处理：{entity}\n{'='*50}")
            try:
                report = self.generate_report(
                    entity, graph,
                    predictor=predictor,
                    feedback_store=feedback_store,
                    weeks=weeks,
                )
                reports[entity] = report
            except Exception as e:
                reports[entity] = f"[错误] {e}"

        # 生成对比摘要
        summary_parts = [f"# 多实体投资对比报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        for entity, report in reports.items():
            # 提取执行摘要部分
            lines = report.split("\n")
            summary = ""
            in_summary = False
            for line in lines:
                if "执行摘要" in line:
                    in_summary = True
                    continue
                if in_summary and line.startswith("---"):
                    break
                if in_summary and line.strip():
                    summary += line + " "
            summary_parts.append(f"## {entity}\n{summary.strip()}\n")

        combined = "\n".join(summary_parts)
        combined += "\n\n---\n*各实体详细报告已分别保存至 reports/ 目录*"
        return combined


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from kg_query import FinDKGGraph
    from kg_predictor import KGPredictor
    from feedback_store import FeedbackStore

    graph     = FinDKGGraph()
    predictor = KGPredictor()
    store     = FeedbackStore()
    orch      = OrchestratorAgent()

    print("\n=== Multi-Agent Tool Use 报告：Apple Inc. ===\n")
    report = orch.generate_report(
        "Apple Inc.", graph,
        predictor=predictor,
        feedback_store=store,
        weeks=12,
    )
    print(report)
