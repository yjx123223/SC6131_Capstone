"""
critic.py
---------
Critic Agent：独立审查 Orchestrator 生成的报告草稿，检查信号冲突、
过度自信、引用准确性。只负责"挑错"，不负责改稿——根据审查意见
修订草稿是 Orchestrator（金融研究员人格）的职责，见
orchestrator_loop.OrchestratorLoop.revise()。

从 orchestrator.py 的 OrchestratorAgent._run_critic 拆分出来。
"""

import json


class CriticAgent:
    """
    严格的风险审查员，独立于 Orchestrator 的 agentic loop。

    使用示例
    --------
    >>> import anthropic
    >>> client = anthropic.Anthropic(api_key="...")
    >>> critic = CriticAgent(client, model="claude-haiku-4-5", max_tokens=1024)
    >>> critique = critic.review(entity, draft, tool_log)
    """

    def __init__(self, client, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def review(self, entity: str, draft: dict, tool_log: list) -> dict:
        """
        审查草稿，检查信号冲突和过度自信。

        Returns
        -------
        {
            "approved": bool,
            "conflicts": list[str],
            "confidence_adjustment": "maintain"|"lower"|"raise",
            "suggestions": str,
        }

        Claude 返回非 JSON 内容或调用异常时，当前策略是默认放行
        （approved=True, conflicts=[]）。这是一个已知的取舍：审查失败
        约等于"没审查"，如果要收紧，可以把默认值改成
        approved=False + confidence_adjustment="lower"，让下游至少
        知道这次审查不可信。目前保留原有行为，未改动。
        """
        context_summary = self._summarize_tool_log(tool_log)

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
                max_tokens=self.max_tokens,
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

    @staticmethod
    def _summarize_tool_log(tool_log: list) -> str:
        """把 OrchestratorLoop 的工具调用记录压缩成给 Critic 看的文字摘要"""
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
        return "\n".join(context_parts) if context_parts else "（无工具调用记录）"
