"""
orchestrator.py
---------------
Orchestrator Agent（改法 A：Tool Use 架构）—— 协调层

架构说明：
  Orchestrator 是一个真正的 Agent——它持有多个工具定义
  （实时行情 / 新闻 / SEC 申报 / 宏观 / 历史评分 / emit_report），
  由 Claude 自主决定调用哪些工具、调用顺序和参数，
  直到调用 emit_report 输出结构化草稿。

  草稿完成后，Critic Agent 独立审查信号冲突与过度自信，
  若发现问题，Orchestrator 进行一轮修订后输出最终报告。

模块划分（本文件只做编排，具体实现拆到各自模块）：
  - orchestrator_loop.OrchestratorLoop  agentic tool-use 循环 + 修订
  - critic.CriticAgent                  独立审查草稿
  - compliance.ComplianceChecker        确定性合规检查 + 置信度兜底
  - report_renderer.render_report       草稿 → Markdown（纯函数）
  - report_store.save_report            保存 Markdown 到本地文件
  - tools/*  各工具的实际业务逻辑
    （FinDKG 图谱工具 tools.kg_tools 已停用，见 orchestrator_loop.py 说明）

OrchestratorAgent 只负责把上面这些部件组装起来，按顺序跑一遍：
  loop.run() → critic.review() → (可选) loop.revise() →
  compliance.check() → render_report() → ensure_disclaimer() → save_report()
"""

from datetime import datetime
from typing import Optional

import anthropic

import config
from orchestrator_loop import OrchestratorLoop
from critic import CriticAgent
from compliance import ComplianceChecker, ensure_disclaimer
from report_renderer import render_report
from report_store import save_report, save_graph_json
from tool_log_summary import latest_result


class OrchestratorAgent:
    """
    Tool Use 架构的 Orchestrator Agent（门面/协调层）。

    使用示例
    --------
    >>> from feedback_store import FeedbackStore
    >>> from orchestrator import OrchestratorAgent
    >>>
    >>> store = FeedbackStore()
    >>> orch  = OrchestratorAgent()
    >>>
    >>> session_id, report = orch.generate_report("Apple Inc.", feedback_store=store)
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
        self.compliance = ComplianceChecker()

    # ── 主入口 ──────────────────────────────────────────────────────

    def generate_report(
        self,
        entity: str,
        graph=None,
        feedback_store=None,
        weeks: int = config.DEFAULT_WEEKS,
    ) -> tuple[Optional[int], str]:
        """
        完整 Multi-Agent 流程：
          Orchestrator loop (tool use) → Critic 审查 → (可选)修订
          → 合规检查与置信度兜底 → 渲染报告 → 保存 → 写入反馈存储

        Parameters
        ----------
        entity         : 目标实体名
        graph          : FinDKGGraph 实例（图谱工具已停用，可不传；保留参数用于兼容）
        feedback_store : FeedbackStore 实例（可选）。传入时会记录本次建议，
                         返回的 session_id 可用于事后评分
        weeks          : 图谱工具的查询周数（已停用，仅记录到反馈存储的 time_window）

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
            reason = getattr(self.loop, "last_stop_reason", None)
            return None, (
                "[Orchestrator] ⚠️ Agent 未能生成报告草稿"
                f"（最后一次 stop_reason={reason}；max_tokens 表示输出被截断，"
                "可调大 config.ORCHESTRATOR_MAX_TOKENS）"
            )

        # 2. Critic Agent 审查
        print(f"\n[Critic] 审查草稿报告...")
        critique = self.critic.review(entity, draft, tool_log)
        approved = critique.get("approved", True)
        conflicts = critique.get("conflicts", [])

        if conflicts:
            print(f"[Critic] 发现 {len(conflicts)} 个问题：" + "；".join(conflicts[:2]))
        else:
            print(f"[Critic] 审查通过，无信号冲突")

        original_confidence = draft.get("confidence")

        # 3. 若 Critic 不通过，进行一轮修订
        if not approved and critique.get("suggestions"):
            print(f"[Orchestrator] 根据 Critic 建议修订报告...")
            draft = self.loop.revise(entity, draft, critique, tool_log)

        # 4. 合规检查 + 置信度兜底（确定性规则，不调用 LLM）
        compliance = self.compliance.check(draft, critique, tool_log, original_confidence)
        draft = compliance["draft"]
        conf = compliance["confidence"]
        if conf["original"] != conf["final"]:
            print(f"[Compliance] 置信度 {conf['original']} → {conf['final']}：{'；'.join(conf['reasons'])}")
        print(
            f"[Compliance] {'通过' if compliance['is_compliant'] else '未通过'}，"
            f"评分 {compliance['score']}/100，问题 {len(compliance['issues'])} 条"
        )

        # 5. 渲染为 Markdown（兜底确认免责声明存在）
        report_md = render_report(entity, draft, critique, tool_log, self.model, compliance=compliance)
        report_md, added = ensure_disclaimer(report_md)
        if added:
            print("[Compliance] ⚠️ 报告缺少免责声明，已自动补充")

        # 6. 保存到本地（报告 + 知识图谱 JSON）
        report_path = save_report(entity, report_md)
        graph_result = latest_result(tool_log, "query_company_graph")
        if graph_result and graph_result.get("_graph"):
            try:
                save_graph_json(report_path, graph_result["_graph"])
            except Exception as e:
                print(f"[Orchestrator] ⚠️  知识图谱保存失败（不影响报告）：{e}")

        # 7. 写入反馈存储（可选），供用户事后评分
        session_id = None
        if feedback_store is not None:
            try:
                session_id = feedback_store.log_advice(
                    entity=entity,
                    kg_summary=self._extract_signal_snapshot(tool_log, draft, compliance),
                    advice_text=report_md,
                    model=self.model,
                    time_window=weeks,
                )
                print(f"[Orchestrator] 建议已记录（session #{session_id}），可事后评分")
            except Exception as e:
                print(f"[Orchestrator] ⚠️  写入反馈存储失败（不影响报告）：{e}")

        return session_id, report_md

    @staticmethod
    def _extract_signal_snapshot(tool_log: list, draft: dict, compliance: Optional[dict] = None) -> dict:
        """
        把本次建议依据的数据快照存档，供事后评分时回看。

        写入 FeedbackStore.log_advice 的 kg_summary 字段（字段名沿用旧表结构）：
          - period       → 行情数据截至日期（advice_sessions.period 列）
          - total_events → 本次成功拿到的新闻条数（advice_sessions.total_events 列）
        新闻全文、SEC 链接等体积较大的内容不存，只存结论相关的关键数值。
        """
        snapshot = {
            "period": "",
            "total_events": 0,
            "recommendation": draft.get("recommendation"),
            "confidence": draft.get("confidence"),
            "news_sentiment": draft.get("news_sentiment"),
            "tools_ok": [],
            "tools_failed": [],
        }
        if compliance:
            snapshot["compliance_score"] = compliance.get("score")
            snapshot["is_compliant"] = compliance.get("is_compliant")
            snapshot["confidence_original"] = compliance.get("confidence", {}).get("original")
        for entry in tool_log:
            tool = entry.get("tool")
            result = entry.get("result") or {}
            if "error" in result:
                snapshot["tools_failed"].append(tool)
                continue
            snapshot["tools_ok"].append(tool)
            if tool == "query_market_data":
                snapshot["ticker"] = result.get("ticker")
                snapshot["period"] = f"行情截至 {result.get('data_as_of')}"
                snapshot["technicals"] = result.get("technicals", {})
                f = result.get("fundamentals", {})
                snapshot["fundamentals"] = {
                    k: f.get(k) for k in ("trailing_pe", "forward_pe", "profit_margin_pct",
                                          "revenue_growth_pct", "analyst_target_mean")
                }
            elif tool == "query_news":
                snapshot["total_events"] = result.get("article_count", 0)
            elif tool == "query_sec_filings":
                snapshot["sec_forms"] = [
                    f"{x.get('form')} {x.get('filing_date')}" for x in result.get("filings", [])
                ]
            elif tool == "query_company_graph":
                snapshot["knowledge_graph"] = {
                    "neighbors": [n["ticker"] for n in result.get("neighbors", [])],
                    "event_count": len(result.get("_event_ids", [])),
                    "top_risks": [
                        {k: r[k] for k in ("event_id", "neighbor", "impact", "score")}
                        for r in result.get("propagated_risks", [])[:3]
                    ],
                    "cited_event_ids": draft.get("cited_event_ids", []),
                }
        return snapshot

    # ── 多实体对比（保持兼容）──────────────────────────────────────

    def generate_comparison_report(
        self,
        entities: list[str],
        graph=None,
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
    from feedback_store import FeedbackStore

    store = FeedbackStore()
    orch  = OrchestratorAgent()

    print("\n=== Multi-Agent Tool Use 报告：Apple Inc. ===\n")
    session_id, report = orch.generate_report("Apple Inc.", feedback_store=store)
    print(report)
    print(f"\n(session #{session_id} 已记录，可用 store.rate({session_id}, +1/0/-1) 评分)")
