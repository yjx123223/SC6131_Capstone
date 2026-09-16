"""
orchestrator.py
---------------
Orchestrator Agent（改法 A：Tool Use 架构）—— 协调层

架构说明：
  Orchestrator 是一个真正的 Agent——它持有4个工具定义，
  由 Claude 自主决定调用哪些工具、调用顺序和参数，
  直到调用 emit_report 输出结构化草稿。

  草稿完成后，Critic Agent 独立审查信号冲突与过度自信，
  若发现问题，Orchestrator 进行一轮修订后输出最终报告。

模块划分（本文件只做编排，具体实现拆到各自模块）：
  - orchestrator_loop.OrchestratorLoop  agentic tool-use 循环 + 修订
  - critic.CriticAgent                  独立审查草稿
  - report_renderer.render_report       草稿 → Markdown（纯函数）
  - report_store.save_report            保存 Markdown 到本地文件
  - tools.kg_tools / tools.macro_tools / tools.feedback_tools
    三个工具的实际业务逻辑，与 mcp_servers/*.py 共用同一份实现

OrchestratorAgent 只负责把上面这些部件组装起来，按顺序跑一遍：
  loop.run() → critic.review() → (可选) loop.revise() →
  render_report() → save_report()
"""

from datetime import datetime
from typing import Optional

import anthropic

import config
from orchestrator_loop import OrchestratorLoop
from critic import CriticAgent
from report_renderer import render_report
from report_store import save_report


class OrchestratorAgent:
    """
    Tool Use 架构的 Orchestrator Agent（门面/协调层）。

    使用示例
    --------
    >>> from kg_query import FinDKGGraph
    >>> from feedback_store import FeedbackStore
    >>> from orchestrator import OrchestratorAgent
    >>>
    >>> graph = FinDKGGraph()
    >>> store = FeedbackStore()
    >>> orch  = OrchestratorAgent()
    >>>
    >>> session_id, report = orch.generate_report("Apple Inc.", graph, feedback_store=store)
    >>> print(report)
    >>> store.rate(session_id, rating=1, note="信号准确")   # 事后评分
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        model: str = config.ORCHESTRATOR_MODEL,
        max_tokens: int = config.ORCHESTRATOR_MAX_TOKENS,
        critic_model: str = config.CRITIC_MODEL,
        critic_max_tokens: int = config.CRITIC_MAX_TOKENS,
    ):
        key = config.get_anthropic_api_key(api_key)
        if not key:
            raise ValueError(
                "未找到 Anthropic API Key。\n"
                "请设置：export ANTHROPIC_API_KEY='your-key'"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

        # loop 和 critic 共用同一个 anthropic client，各自独立配置模型/预算
        self.loop = OrchestratorLoop(
            self.client, model, max_tokens,
            fred_api_key=config.get_fred_api_key(fred_api_key),
        )
        self.critic = CriticAgent(self.client, critic_model, critic_max_tokens)

    # ── 主入口 ──────────────────────────────────────────────────────

    def generate_report(
        self,
        entity: str,
        graph,
        feedback_store=None,
        weeks: int = config.DEFAULT_WEEKS,
    ) -> tuple[Optional[int], str]:
        """
        完整 Multi-Agent 流程：
          Orchestrator loop (tool use) → Critic 审查 → (可选)修订 → 渲染报告
          → 保存 → 写入反馈存储

        Parameters
        ----------
        entity         : 目标实体名
        graph          : FinDKGGraph 实例
        feedback_store : FeedbackStore 实例（可选）。传入时会记录本次建议，
                         返回的 session_id 可用于事后评分
        weeks          : 查询最近多少周

        Returns
        -------
        (session_id, report_md)
          session_id : 供 FeedbackStore.rate() 事后评分；未传 feedback_store
                       或草稿生成失败时为 None
          report_md  : Markdown 格式的最终投资建议报告

        说明：返回值格式与单 Agent 链路的 AssetAdvisor.advise() 保持一致，
        两条链路的评分入口因此可以复用同一套交互逻辑。
        """
        # 1. Orchestrator agentic loop
        draft, tool_log = self.loop.run(
            entity, weeks, graph, feedback_store=feedback_store
        )

        if draft is None:
            return None, f"[Orchestrator] ⚠️ Agent 未能生成报告草稿（超出最大迭代次数或异常退出）"

        # 2. Critic Agent 审查
        print(f"\n[Critic] 审查草稿报告...")
        critique = self.critic.review(entity, draft, tool_log)
        approved = critique.get("approved", True)
        conflicts = critique.get("conflicts", [])

        if conflicts:
            print(f"[Critic] 发现 {len(conflicts)} 个问题：" + "；".join(conflicts[:2]))
        else:
            print(f"[Critic] 审查通过，无信号冲突")

        # 3. 若 Critic 不通过，进行一轮修订
        if not approved and critique.get("suggestions"):
            print(f"[Orchestrator] 根据 Critic 建议修订报告...")
            draft = self.loop.revise(entity, draft, critique, tool_log)

        # 4. 渲染为 Markdown
        report_md = render_report(entity, draft, critique, tool_log, self.model)

        # 5. 保存到本地
        save_report(entity, report_md)

        # 6. 写入反馈存储（可选），供用户事后评分
        session_id = None
        if feedback_store is not None:
            try:
                session_id = feedback_store.log_advice(
                    entity=entity,
                    kg_summary=self._extract_kg_summary(tool_log),
                    advice_text=report_md,
                    model=self.model,
                    time_window=weeks,
                )
                print(f"[Orchestrator] 建议已记录（session #{session_id}），可事后评分")
            except Exception as e:
                print(f"[Orchestrator] ⚠️  写入反馈存储失败（不影响报告）：{e}")

        return session_id, report_md

    @staticmethod
    def _extract_kg_summary(tool_log: list) -> dict:
        """
        从工具调用记录里取出 KG 信号快照，作为本次建议的依据存档。

        FeedbackStore.signal_accuracy_report() 依赖 kg_summary 里的
        positive_impacts / negative_impacts（每条含 relation 字段）来按
        关系类型聚合历史评分，因此这里必须存 query_kg_signals 的原始结果。
        多跳上下文（multihop_context）体积较大且不参与关系类型统计，剔除。
        """
        for entry in tool_log:
            if entry.get("tool") != "query_kg_signals":
                continue
            result = entry.get("result") or {}
            if "error" in result:
                continue
            return {k: v for k, v in result.items() if k != "multihop_context"}
        return {}

    # ── 多实体对比（保持兼容）──────────────────────────────────────

    def generate_comparison_report(
        self,
        entities: list[str],
        graph,
        feedback_store=None,
        weeks: int = config.DEFAULT_WEEKS,
    ) -> str:
        """
        多实体对比：分别为每个实体运行完整 Agent 流程，最后合并对比摘要。
        """
        reports = {}
        for entity in entities:
            print(f"\n{'='*50}\n处理：{entity}\n{'='*50}")
            try:
                _session_id, report = self.generate_report(
                    entity, graph,
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
    from feedback_store import FeedbackStore

    graph = FinDKGGraph()
    store = FeedbackStore()
    orch  = OrchestratorAgent()

    print("\n=== Multi-Agent Tool Use 报告：Apple Inc. ===\n")
    session_id, report = orch.generate_report(
        "Apple Inc.", graph,
        feedback_store=store,
        weeks=config.DEFAULT_WEEKS,
    )
    print(report)
    print(f"\n(session #{session_id} 已记录，可用 store.rate({session_id}, +1/0/-1) 评分)")
