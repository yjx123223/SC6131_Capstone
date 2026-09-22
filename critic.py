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

from report_fields import format_draft
from tool_log_summary import summarize_tool_log


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
        context_summary = summarize_tool_log(tool_log)

        system_prompt = """你是一位严格的风险审查员（Critic Agent）。
你将收到一份投资建议报告草稿和生成该报告所用的原始数据摘要。

检查以下五点：
1. 信号冲突：宏观、基本面/估值、技术面、新闻情绪之间方向是否矛盾？若有冲突，报告是否明确标注并说明取舍？
2. 过度自信：置信度（high/medium/low）是否与数据完整度匹配？有工具返回"数据不可用"时不应标为 high。
3. 引用准确性：报告中的数值、新闻、申报是否都能在原始数据摘要中找到？是否存在编造或无依据的推断？
4. 数据时效：报告是否基于摘要中的最新数据日期，有没有把旧信息当成最新情况？
5. 知识图谱引用：报告引用的事件编号（E1、E2…）是否都出现在摘要中？对传导风险的描述是否与路径一致？是否把基于规则的推断说成了确定事实？

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
{format_draft(draft)}

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
        """兼容旧调用方：实现已移至 tool_log_summary.summarize_tool_log"""
        return summarize_tool_log(tool_log)
