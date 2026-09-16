"""
orchestrator_loop.py
----------------------
OrchestratorLoop：Orchestrator Agent 的 agentic tool-use 循环。

职责：
  - 持有工具的 Anthropic tool_use schema 定义（实时行情 / 新闻 / SEC 申报 /
    宏观 / 历史评分 / emit_report）
  - 驱动"Claude 自主决定调用哪些工具、调用顺序和参数"的多轮循环，
    直到调用 emit_report 结束
  - 把工具调用分发到 tools/ 下的各工具模块（实际业务逻辑在那边，
    这里只做参数组装）
  - revise()：Critic 审查不通过时，以"金融研究员"人格根据审查意见
    修订草稿（复用同一份 emit_report schema 强制结构化输出）

feat/market 变更：
  - 新增 query_market_data / query_news / query_sec_filings 三个实时数据工具
  - query_kg_signals 已注释停用：FinDKG 数据截止 2023-01-01，与实时行情
    存在时间错位。tools/kg_tools.py 本身保留未删除，需要恢复时取消下方
    注释即可
  - emit_report 字段改为基本面 / 技术面 / 新闻舆情 / 监管申报（见 report_fields.py）

不包含：Critic 审查逻辑（见 critic.CriticAgent）、报告渲染
（见 report_renderer.render_report）、报告保存（见 report_store.save_report）。

从 orchestrator.py 的 OrchestratorAgent 拆分出来。
"""

import json
from typing import Optional

import config
from report_fields import CONFIDENCE_ENUM, RECOMMENDATION_ENUM, SENTIMENT_ENUM, format_draft
from tool_log_summary import summarize_tool_log
from tools import macro_tools, feedback_tools, market_tools, news_tools, sec_tools
# from tools import kg_tools   # FinDKG 图谱工具已停用，见模块说明


# ── Tool 定义 ────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    # ── FinDKG 图谱工具（已停用：数据截止 2023-01-01，与实时数据时间错位）──
    # {
    #     "name": "query_kg_signals",
    #     "description": (
    #         "查询目标实体在 FinDKG 知识图谱中的历史事件信号，"
    #         "包括正面影响事件（Positive_Impact_On / Raise / Invests_In）、"
    #         "负面影响事件（Negative_Impact_On / Decrease）、其他关联事件。"
    #     ),
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "entity": {
    #                 "type": "string",
    #                 "description": "实体名称，如 'Apple Inc.'",
    #             },
    #             "weeks": {
    #                 "type": "integer",
    #                 "description": "查询最近 N 周，默认 12",
    #                 "default": 12,
    #             },
    #         },
    #         "required": ["entity"],
    #     },
    # },
    {
        "name": "query_market_data",
        "description": (
            "通过 Yahoo Finance 获取目标公司的实时市场数据：公司概况、估值与财务指标"
            "（市值、PE、利润率、营收增速、ROE、负债率、分析师目标价等）、"
            "近期收盘价，以及确定性计算的技术指标（MA20/MA50、RSI14、波动率、涨跌幅）。"
            f"数据最新交易日超过 {config.MARKET_MAX_STALENESS_DAYS} 天会返回 error。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "公司名或股票代码，如 'Apple Inc.' 或 'AAPL'",
                },
                "period": {
                    "type": "string",
                    "enum": list(config.MARKET_ALLOWED_PERIODS),
                    "description": f"行情回看窗口，默认 {config.MARKET_DEFAULT_PERIOD}",
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "query_news",
        "description": (
            f"获取目标公司最近 {config.NEWS_LOOKBACK_DAYS} 天内的新闻（标题、来源、发布时间、摘要、链接），"
            "按时间倒序。你需要自行阅读并判断整体新闻情绪。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "公司名或股票代码",
                },
                "max_items": {
                    "type": "integer",
                    "description": f"最多返回条数，默认 {config.NEWS_MAX_ITEMS}，上限 20",
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "query_sec_filings",
        "description": (
            f"从 SEC EDGAR 获取目标公司最近 {config.SEC_LOOKBACK_DAYS} 天内的监管申报列表"
            "（10-K 年报、10-Q 季报、8-K 重大事项等），含申报日期与原文链接。"
            "仅适用于在 SEC 申报的公司。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "公司名或股票代码",
                },
                "form_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "申报类型过滤，如 ['10-K', '8-K']，默认 10-K/10-Q/8-K/20-F/6-K",
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
            "某项数据工具返回 error 时，对应字段写明'数据不可用'及原因，不要编造。"
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
                    "description": "宏观环境分析（利率、通胀、就业、VIX）",
                },
                "fundamental_analysis": {
                    "type": "string",
                    "description": "基本面与估值分析（引用具体指标数值）",
                },
                "technical_analysis": {
                    "type": "string",
                    "description": "技术面分析（趋势、均线、RSI、波动率，注明数据截至日期）",
                },
                "news_sentiment_analysis": {
                    "type": "string",
                    "description": "近期新闻舆情分析（引用具体新闻标题与日期）",
                },
                "news_sentiment": {
                    "type": "string",
                    "enum": SENTIMENT_ENUM,
                    "description": "整体新闻情绪；新闻不可用时填 unavailable",
                },
                "filings_analysis": {
                    "type": "string",
                    "description": "近期 SEC 申报要点（有哪些申报、时间、可能含义）",
                },
                "recommendation": {
                    "type": "string",
                    "enum": RECOMMENDATION_ENUM,
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
                    "enum": CONFIDENCE_ENUM,
                    "description": "整体置信度",
                },
                "key_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "支撑结论的关键信号列表（3-5条，每条注明数据来源）",
                },
            },
            "required": [
                "executive_summary",
                "macro_analysis",
                "fundamental_analysis",
                "technical_analysis",
                "news_sentiment_analysis",
                "news_sentiment",
                "filings_analysis",
                "recommendation",
                "recommendation_rationale",
                "risk_warnings",
                "confidence",
                "key_signals",
            ],
        },
    },
]


SYSTEM_PROMPT = """你是一位专业的金融研究员 Agent。
你的任务是基于最新的真实数据，为指定公司生成投资建议报告。

你有以下工具可以使用：
- query_market_data：实时公司概况、估值/财务指标、近期行情与技术指标
- query_news：近两周的公司新闻
- query_sec_filings：近一年的 SEC 监管申报（10-K / 10-Q / 8-K 等）
- query_macro：当前宏观经济指标（利率、通胀、失业率、VIX）
- get_feedback_stats：历史建议的用户评分统计（可选）

工作流程建议（你可以根据情况调整，可以在同一轮并行调用多个工具）：
1. 查询市场数据，掌握基本面、估值与技术面
2. 查询近期新闻，自行判断新闻情绪，并关注重大事件
3. 查询 SEC 申报，确认近期财报 / 重大事项披露
4. 查询宏观指标，判断整体市场环境
5. 信息足够时，调用 emit_report 输出结构化报告

注意：
- 只使用工具返回的数据，引用数值时注明数据截至日期；不要编造数字、新闻或申报
- 某个工具返回 error 时，不要反复重试同一调用；在报告对应部分写明"数据不可用"及原因，并相应降低置信度
- 宏观信号决定整体仓位方向，公司基本面/技术面/新闻决定个股判断
- 若各类信号方向相反（如基本面强但技术面走弱、宏观偏空），在报告中明确标注冲突并说明如何取舍
- emit_report 的 confidence 需真实反映数据完整度与信号一致性，不要过度自信"""


class OrchestratorLoop:
    """
    Orchestrator 的 agentic tool-use 循环。

    使用示例
    --------
    >>> import anthropic
    >>> client = anthropic.Anthropic(api_key="...")
    >>> loop = OrchestratorLoop(client, model="claude-haiku-4-5", max_tokens=2048, fred_api_key="...")
    >>> draft, tool_log = loop.run("Apple Inc.", feedback_store=store)
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
        weeks: int = config.DEFAULT_WEEKS,
        graph=None,
        feedback_store=None,
    ) -> tuple[Optional[dict], list]:
        """
        跑一次完整的 agentic loop，直到调用 emit_report 或耗尽 MAX_ITERATIONS。

        weeks / graph 是 FinDKG 图谱工具的参数，图谱工具停用后不再使用，
        保留参数只为兼容现有调用方（恢复图谱工具时无需改签名）。

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

        system_prompt = SYSTEM_PROMPT
        today = self._today()
        initial_message = (
            f"今天是 {today}。请为 [{entity}] 生成投资建议报告，"
            f"基于最新的市场、新闻、监管申报与宏观数据。"
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
                        "_is_error": "error" in result,
                    })

            # 构建标准 tool_result 消息
            content_blocks = []
            for tr in tool_results:
                block = {
                    "type":        "tool_result",
                    "tool_use_id": tr["_id"],
                    "content":     tr["_content"],
                }
                if tr.get("_is_error"):
                    block["is_error"] = True   # 让模型明确知道这次工具调用失败了
                content_blocks.append(block)
            messages.append({"role": "user", "content": content_blocks})

            if emit_called:
                break

        return draft, tool_log

    # ── 工具执行器 ───────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        from tools.yf_client import utc_now
        return utc_now().date().isoformat()

    def _execute_tool(self, name: str, tool_input: dict, context: dict) -> dict:
        """根据工具名分发执行（实际业务逻辑在 tools/ 模块）"""
        handlers = {
            # "query_kg_signals": lambda: self._tool_query_kg(tool_input, context),   # 图谱工具已停用
            "query_market_data":  lambda: self._tool_query_market(tool_input),
            "query_news":         lambda: self._tool_query_news(tool_input),
            "query_sec_filings":  lambda: self._tool_query_sec(tool_input),
            "query_macro":        lambda: self._tool_query_macro(tool_input),
            "get_feedback_stats": lambda: self._tool_feedback_stats(tool_input, context),
        }
        handler = handlers.get(name)
        if handler is None:
            result = {"error": f"未知工具：{name}"}
        else:
            try:
                result = handler()
            except Exception as e:   # 工具层约定不抛异常，这里兜底防止整个 loop 崩溃
                result = {"error": f"{name} 执行异常：{e}"}

        if "error" in result:
            print(f"[Orchestrator] ⚠️  {name} 错误：{result['error']}")

        return result

    # FinDKG 图谱工具已停用（数据截止 2023-01-01，与实时数据时间错位）
    # def _tool_query_kg(self, tool_input: dict, context: dict) -> dict:
    #     graph  = context["graph"]
    #     entity = tool_input.get("entity", "")
    #     weeks  = tool_input.get("weeks", context.get("default_weeks", config.DEFAULT_WEEKS))
    #     return kg_tools.query_kg_signals(entity, weeks=weeks, graph=graph)

    def _tool_query_market(self, tool_input: dict) -> dict:
        return market_tools.query_market_data(
            tool_input.get("entity", ""),
            period=tool_input.get("period") or config.MARKET_DEFAULT_PERIOD,
        )

    def _tool_query_news(self, tool_input: dict) -> dict:
        return news_tools.query_news(
            tool_input.get("entity", ""),
            max_items=tool_input.get("max_items") or config.NEWS_MAX_ITEMS,
        )

    def _tool_query_sec(self, tool_input: dict) -> dict:
        return sec_tools.query_sec_filings(
            tool_input.get("entity", ""),
            form_types=tool_input.get("form_types"),
        )

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
{format_draft(draft)}

【原始数据摘要】
{summarize_tool_log(tool_log)}

【Critic 发现的问题】
{conflicts_text}

【修改建议】
{suggestions}

请根据以上问题修订报告（只能使用原始数据摘要中的信息，不要编造），然后调用 emit_report 输出修订后的完整版本。"""

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
