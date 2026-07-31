"""
orchestrator_loop.py
----------------------
OrchestratorLoop：Orchestrator Agent 的 agentic tool-use 循环。

职责：
  - 持有 4 个工具的 Anthropic tool_use schema 定义
  - 驱动"Claude 自主决定调用哪些工具、调用顺序和参数"的多轮循环，
    直到调用 emit_report 结束
  - 把工具调用分发到 tools.kg_tools / tools.macro_tools /
    tools.feedback_tools（实际业务逻辑在那边，这里只做参数组装）
  - revise()：Critic 审查不通过时，以"金融研究员"人格根据审查意见
    修订草稿（复用同一份 emit_report schema 强制结构化输出）

不包含：Critic 审查逻辑（见 critic.CriticAgent）、报告渲染
（见 report_renderer.render_report）、报告保存（见 report_store.save_report）。

从 orchestrator.py 的 OrchestratorAgent 拆分出来。
"""

import json
from typing import Optional

from tools import kg_tools, macro_tools, feedback_tools


# ── Tool 定义 ────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "query_kg_signals",
        "description": (
            "查询目标实体在 FinDKG 知识图谱中的历史事件信号，"
            "包括正面影响事件（Positive_Impact_On / Raise / Invests_In）、"
            "负面影响事件（Negative_Impact_On / Decrease）、其他关联事件。"
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


class OrchestratorLoop:
    """
    Orchestrator 的 agentic tool-use 循环。

    使用示例
    --------
    >>> import anthropic
    >>> client = anthropic.Anthropic(api_key="...")
    >>> loop = OrchestratorLoop(client, model="claude-haiku-4-5", max_tokens=2048, fred_api_key="...")
    >>> draft, tool_log = loop.run("Apple Inc.", weeks=12, graph=graph, feedback_store=store)
    """

    MAX_ITERATIONS = 10   # agentic loop 最大轮次（防止无限循环）

    def __init__(self, client, model: str, max_tokens: int, fred_api_key: Optional[str] = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fred_api_key = fred_api_key

    # ── 主入口 ──────────────────────────────────────────────────────

    def run(
        self,
        entity: str,
        weeks: int,
        graph,
        feedback_store=None,
    ) -> tuple[Optional[dict], list]:
        """
        跑一次完整的 agentic loop，直到调用 emit_report 或耗尽 MAX_ITERATIONS。

        Returns
        -------
        (draft_dict_or_None, tool_call_log)
        """
        context = {
            "graph": graph,
            "feedback_store": feedback_store,
            "default_weeks": weeks,
        }

        print(f"\n[Orchestrator] 启动 Agent Loop — 目标实体：{entity}")

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
                tools=TOOL_DEFINITIONS,
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
                        "_id": tool_id,
                        "_content": json.dumps(result, ensure_ascii=False, default=str),
                    })

            # 构建标准 tool_result 消息
            content_blocks = [
                {
                    "type":        "tool_result",
                    "tool_use_id": tr["_id"],
                    "content":     tr["_content"],
                }
                for tr in tool_results
            ]
            messages.append({"role": "user", "content": content_blocks})

            if emit_called:
                break

        return draft, tool_log

    # ── 工具执行器 ───────────────────────────────────────────────────

    def _execute_tool(self, name: str, tool_input: dict, context: dict) -> dict:
        """根据工具名分发执行（实际业务逻辑在 tools/ 模块）"""
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
        import config

        graph  = context["graph"]
        entity = tool_input.get("entity", "")
        weeks  = tool_input.get("weeks", context.get("default_weeks", config.DEFAULT_WEEKS))

        return kg_tools.query_kg_signals(entity, weeks=weeks, graph=graph)

    def _tool_query_macro(self, tool_input: dict) -> dict:
        return macro_tools.query_macro(self.fred_api_key, indicators=tool_input.get("indicators"))

    def _tool_feedback_stats(self, tool_input: dict, context: dict) -> dict:
        store = context.get("feedback_store")
        if store is None:
            return {"error": "未配置 FeedbackStore，历史评分不可用"}

        return feedback_tools.get_feedback_stats(
            relation_type=tool_input.get("relation_type", ""), store=store
        )

    # ── 修订（以 Orchestrator/金融研究员人格进行）────────────────────

    def revise(self, entity: str, draft: dict, critique: dict, tool_log: list) -> dict:
        """
        Critic 不通过时进行修订。以"金融研究员"人格（跟 run() 同一个
        Orchestrator 身份，而不是 Critic 身份）根据审查意见改稿。
        使用 tool_use 强制调用 emit_report，避免 JSON 转义问题。
        """
        conflicts_text = "\n".join(f"- {c}" for c in critique.get("conflicts", []))
        suggestions    = critique.get("suggestions", "")

        emit_tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "emit_report")

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
