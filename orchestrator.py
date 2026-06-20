"""
orchestrator.py
---------------
Orchestrator Agent：汇总 News Agent 和 Macro Agent 的信号，
调用 Claude 生成最终投资建议报告。

优先级原则（方案 A）：
  - 宏观信号决定整体仓位方向
  - 新闻（KG）信号决定个股选择与具体建议
  - 若宏观偏空，即使新闻正面也不建议超配

输出：结构化的 Markdown 格式投资建议报告
"""

import os
from typing import Optional
import anthropic


class OrchestratorAgent:
    """
    Multi-Agent 汇总器：整合宏观 + 新闻信号，生成投资建议报告。

    使用示例
    --------
    >>> from macro_agent import MacroAgent
    >>> from llm_advisor import AssetAdvisor
    >>> from kg_query import FinDKGGraph
    >>> from orchestrator import OrchestratorAgent
    >>>
    >>> graph = FinDKGGraph()
    >>> news_agent = AssetAdvisor()
    >>> macro_agent = MacroAgent()
    >>> orch = OrchestratorAgent()
    >>>
    >>> news_signal = news_agent.get_news_signal("Apple Inc.", graph)
    >>> macro_signal = macro_agent.analyze()
    >>> report = orch.generate_report("Apple Inc.", news_signal, macro_signal)
    >>> print(report)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
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

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_prompt(
        self,
        entity: str,
        news_signal: dict,
        macro_signal: dict,
    ) -> tuple[str, str]:
        """将两路信号组合为 Orchestrator prompt"""

        # 宏观信号文本
        direction_zh = {
            "bullish": "偏多（风险资产友好）",
            "bearish": "偏空（风险资产承压）",
            "neutral": "中性（方向不明）",
        }.get(macro_signal.get("direction", "neutral"), "中性")

        confidence_zh = {
            "high": "高", "medium": "中", "low": "低"
        }.get(macro_signal.get("confidence", "low"), "低")

        macro_block = f"""【宏观环境信号 — Macro Agent】
整体方向：{direction_zh}
置信度：{confidence_zh}
判断理由：{macro_signal.get('rationale', '无')}

关键指标：
{macro_signal.get('summary_text', '（数据不可用）')}"""

        # 新闻信号文本
        if "error" in news_signal:
            news_block = f"【新闻知识图谱信号 — News Agent】\n错误：{news_signal['error']}"
        else:
            def fmt_events(events: list) -> str:
                if not events:
                    return "  （无）"
                return "\n".join(
                    f"  - {e['subject']} --[{e['relation']}]--> {e['object']}  (出现{e['count']}次)"
                    for e in events[:6]
                )

            pred_block = ""
            if news_signal.get("has_predictions") and news_signal.get("predictions"):
                preds = news_signal["predictions"]
                pred_lines = "\n".join(
                    f"  - {p['subject']} → {p['object']}  (置信度 {p['score']:.3f})"
                    for p in preds[:5]
                )
                pred_block = f"\nKGTransformer 预测信号（下一周，仅供参考）：\n{pred_lines}"

            news_block = f"""【新闻知识图谱信号 — News Agent (FinDKG / WSJ)】
目标实体：{news_signal['entity']}
数据时间段：{news_signal.get('period', '未知')}
事件总数：{news_signal.get('total_events', 0)}

正面影响事件：
{fmt_events(news_signal.get('positive_impacts', []))}

负面影响事件：
{fmt_events(news_signal.get('negative_impacts', []))}

其他关联事件：
{fmt_events(news_signal.get('other_relations', []))}{pred_block}"""

        system_prompt = """你是一位资深投资顾问，负责整合多维度信号生成投资建议报告。

你将收到两类信号：
1. 宏观环境信号（Macro Agent）：反映整体市场方向
2. 新闻知识图谱信号（News Agent）：反映个股/实体的具体事件

【优先级原则】
- 宏观信号决定整体仓位方向：偏空时不超配风险资产，偏多时可适当积极
- 新闻信号决定个股选择：在宏观允许的范围内，正面新闻信号支持增持，负面信号建议减仓
- 若两类信号方向相反，以宏观为主，个股建议须附加风险提示

【报告格式要求】
输出 Markdown 格式报告，包含以下章节：
1. 执行摘要（2-3句话，核心结论）
2. 宏观环境分析
3. 个股信号分析（基于新闻KG数据）
4. 综合配置建议（持有/增持/减仓/观望，并说明理由）
5. 主要风险提示

结尾声明：本报告由 AI 系统自动生成，仅供参考，不构成投资建议。"""

        user_prompt = f"""{macro_block}

---

{news_block}

---

请基于以上两路信号，为 [{entity}] 生成一份结构化投资建议报告。"""

        return system_prompt, user_prompt

    # ── 核心生成 ────────────────────────────────────────────────────

    def generate_report(
        self,
        entity: str,
        news_signal: dict,
        macro_signal: dict,
    ) -> str:
        """
        生成最终投资建议报告。

        Parameters
        ----------
        entity       : 目标实体名
        news_signal  : NewsAgent.get_news_signal() 的返回值
        macro_signal : MacroAgent.analyze() 的返回值

        Returns
        -------
        Markdown 格式的投资建议报告文本
        """
        print(f"[Orchestrator] 整合信号，生成 [{entity}] 投资建议报告...")

        system_prompt, user_prompt = self._build_prompt(entity, news_signal, macro_signal)

        print(f"[Orchestrator] 调用 {self.model}...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        report_text = response.content[0].text
        usage = response.usage

        # 报告页脚
        macro_dir = macro_signal.get("direction", "unknown")
        news_events = news_signal.get("total_events", 0)
        has_pred = "有" if news_signal.get("has_predictions") else "无"

        footer = (
            f"\n\n---\n"
            f"*生成模型：{self.model} | "
            f"宏观方向：{macro_dir} | "
            f"KG事件数：{news_events} | "
            f"预测信号：{has_pred} | "
            f"Token：{usage.input_tokens} in / {usage.output_tokens} out*"
        )

        return report_text + footer

    def generate_comparison_report(
        self,
        entities: list[str],
        news_signals: list[dict],
        macro_signal: dict,
    ) -> str:
        """
        多实体对比报告：同一宏观背景下对多个实体做横向比较。

        Parameters
        ----------
        entities      : 实体名列表
        news_signals  : 与 entities 对应的 news_signal 列表
        macro_signal  : 共用的宏观信号
        """
        print(f"[Orchestrator] 生成多实体对比报告：{', '.join(entities)}...")

        direction_zh = {
            "bullish": "偏多", "bearish": "偏空", "neutral": "中性"
        }.get(macro_signal.get("direction", "neutral"), "中性")

        macro_block = f"""【宏观环境】
方向：{direction_zh} | 置信度：{macro_signal.get('confidence', 'low')}
{macro_signal.get('rationale', '')}

指标：
{macro_signal.get('summary_text', '')}"""

        # 每个实体的信号摘要
        entity_blocks = []
        for entity, ns in zip(entities, news_signals):
            if "error" in ns:
                entity_blocks.append(f"**{entity}**：数据不可用（{ns['error']}）")
                continue
            pos_count = len(ns.get("positive_impacts", []))
            neg_count = len(ns.get("negative_impacts", []))
            top_pos = ns["positive_impacts"][0] if ns.get("positive_impacts") else None
            top_neg = ns["negative_impacts"][0] if ns.get("negative_impacts") else None
            block = (
                f"**{entity}**：事件总数={ns['total_events']} | "
                f"正面类别={pos_count} | 负面类别={neg_count}\n"
            )
            if top_pos:
                block += f"  最强正面：{top_pos['subject']} --[{top_pos['relation']}]--> {top_pos['object']} (×{top_pos['count']})\n"
            if top_neg:
                block += f"  最强负面：{top_neg['subject']} --[{top_neg['relation']}]--> {top_neg['object']} (×{top_neg['count']})\n"
            entity_blocks.append(block)

        entity_list_str = "、".join(entities)
        signals_block = "\n".join(entity_blocks)

        system_prompt = "你是一位资深投资顾问，擅长在宏观背景下对多个资产进行横向比较分析。"
        user_prompt = f"""{macro_block}

---

【各实体新闻KG信号摘要】
{signals_block}

请对以上实体（{entity_list_str}）进行横向对比分析，给出：
1. 宏观背景下的整体配置建议
2. 各实体信号强弱排序与理由
3. 综合建议（超配/标配/低配/回避）
4. 主要风险点

结尾声明：本报告由 AI 系统自动生成，仅供参考，不构成投资建议。"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from kg_query import FinDKGGraph
    from kg_predictor import KGPredictor
    from llm_advisor import AssetAdvisor
    from macro_agent import MacroAgent

    graph     = FinDKGGraph()
    predictor = KGPredictor()
    news_agent  = AssetAdvisor(predictor=predictor)
    macro_agent = MacroAgent()
    orch        = OrchestratorAgent()

    entity = "Apple Inc."

    print(f"\n=== Multi-Agent 投资报告：{entity} ===\n")
    news_signal  = news_agent.get_news_signal(entity, graph, n_recent_weeks=12)
    macro_signal = macro_agent.analyze()
    report       = orch.generate_report(entity, news_signal, macro_signal)

    print("\n" + "="*60)
    print(report)
