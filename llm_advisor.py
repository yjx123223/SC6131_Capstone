"""
llm_advisor.py
--------------
将 KG 查询结果构建为 prompt，调用 Claude API 生成资产配置建议。
支持可选接入 KGTransformer 预测信号和 FeedbackStore 反馈存储。
"""

from typing import Optional
import anthropic

import config


class AssetAdvisor:
    """
    基于 FinDKG 知识图谱上下文，调用 Claude 生成资产配置建议。

    使用示例
    --------
    >>> from kg_query import FinDKGGraph
    >>> from llm_advisor import AssetAdvisor
    >>>
    >>> graph = FinDKGGraph()
    >>> advisor = AssetAdvisor()
    >>> session_id, advice = advisor.advise("Apple Inc.", graph, n_recent_weeks=12)
    >>> print(advice)
    >>> advisor.feedback(session_id, rating=1)   # 事后评分
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = config.ADVISOR_MODEL,
        max_tokens: int = config.ADVISOR_MAX_TOKENS,
        predictor=None,       # KGPredictor 实例（可选）
        feedback_store=None,  # FeedbackStore 实例（可选）
    ):
        """
        Parameters
        ----------
        api_key        : Anthropic API Key。未传入则从环境变量 ANTHROPIC_API_KEY 读取。
        model          : 使用的 Claude 模型，默认取 config.ADVISOR_MODEL。
        max_tokens     : 回复最大 token 数，默认取 config.ADVISOR_MAX_TOKENS。
        predictor      : KGPredictor 实例。传入后会将预测信号追加到 prompt。
        feedback_store : FeedbackStore 实例。传入后每次建议自动写入记录。
        """
        key = config.get_anthropic_api_key(api_key)
        if not key:
            raise ValueError(
                "未找到 Anthropic API Key。\n"
                "请设置环境变量：export ANTHROPIC_API_KEY='your-key'\n"
                "或在初始化时传入：AssetAdvisor(api_key='your-key')"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.predictor = predictor
        self.feedback_store = feedback_store

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_prompt(
        self,
        entity: str,
        summary: dict,
        user_question: str,
        predictions: list[dict] | None = None,
    ) -> tuple[str, str]:
        """将 KG 摘要（+可选预测信号）结构化为 LLM prompt"""

        pos = summary.get("positive_impacts", [])
        neg = summary.get("negative_impacts", [])
        other = summary.get("other_relations", [])

        def fmt_events(events: list[dict]) -> str:
            if not events:
                return "（无）"
            lines = [
                f"  - {e['subject']} --[{e['relation']}]--> {e['object']}  (出现{e['count']}次)"
                for e in events[:config.MAX_EVENTS_IN_PROMPT]
            ]
            return "\n".join(lines)

        def fmt_predictions(preds: list[dict]) -> str:
            if not preds:
                return "（无预测数据，KGTransformer 尚未训练）"
            lines = [
                f"  - {p['subject']} → {p['object']}  (置信度 {p['score']:.3f})"
                for p in preds[:config.MAX_EVENTS_IN_PROMPT]
            ]
            return "\n".join(lines)

        # 预测信号区块（有 predictor 且有结果才显示）
        prediction_block = ""
        if predictions is not None:
            prediction_block = f"""

【KGTransformer 预测信号（下一周，仅供参考）】
{fmt_predictions(predictions)}
注意：以上为模型预测，非历史事实，请与历史信号结合判断。"""

        kg_context = f"""
【知识图谱上下文 — 历史事实】
目标实体：{entity}
数据时间段：{summary.get('period', '未知')}
事件总数：{summary.get('total_events', 0)}

正面影响事件（Positive_Impact_On / Raise / Invests_In）：
{fmt_events(pos)}

负面影响事件（Negative_Impact_On / Decrease）：
{fmt_events(neg)}

其他关联事件：
{fmt_events(other)}{prediction_block}
""".strip()

        system_prompt = """你是一位专业的金融分析师助手。
你将收到来自金融知识图谱（FinDKG）的结构化事件数据，这些数据反映了金融实体之间的真实历史关联关系，以及（如有）KGTransformer 模型对下一时间步的预测信号。
请基于这些数据，结合金融分析逻辑，为用户提供资产配置参考建议。

注意事项：
1. 历史事实与模型预测需明确区分，预测信号权重低于历史事实
2. 明确区分"正面信号"与"负面信号"
3. 建议需包含：信号分析 → 配置建议 → 风险提示
4. 结尾声明：本建议仅供参考，不构成投资意见"""

        user_prompt = f"""{kg_context}

用户问题：{user_question}

请基于以上知识图谱数据，提供结构化的资产配置分析与建议。"""

        return system_prompt, user_prompt

    # ── 核心调用 ─────────────────────────────────────────────────

    def advise(
        self,
        entity: str,
        graph,
        n_recent_weeks: int = config.DEFAULT_WEEKS,
        user_question: Optional[str] = None,
    ) -> tuple[Optional[int], str]:
        """
        完整流程：KG查询 → (可选)预测 → prompt构建 → Claude API → (可选)写入反馈 → 返回建议

        Parameters
        ----------
        entity          : 查询实体名（如 "Apple Inc."）
        graph           : FinDKGGraph 实例
        n_recent_weeks  : 使用最近多少周数据
        user_question   : 用户的具体问题（可选）

        Returns
        -------
        (session_id, advice_text)
          session_id 可传入 feedback() 进行事后评分；若无 FeedbackStore 则为 None
        """
        # 1. KG 历史查询
        print(f"[Advisor] 查询知识图谱：{entity}（最近{n_recent_weeks}周）...")
        summary = graph.get_impact_summary(entity, n_recent_weeks=n_recent_weeks)

        if summary.get("total_events", 0) == 0:
            return None, f"知识图谱中未找到 '{entity}' 在最近 {n_recent_weeks} 周内的事件数据。"

        # 2. KGTransformer 预测（可选）
        predictions = None
        if self.predictor and self.predictor.available:
            print(f"[Advisor] 运行 KGTransformer 预测...")
            predictions = self.predictor.predict_next(entity, graph, top_k=10)

        # 3. 构建 prompt
        default_question = (
            f"基于知识图谱中 {entity} 的近期事件，"
            f"请分析其正负面信号，并给出资产配置建议（持有/减仓/增持/观望）。"
        )
        question = user_question or default_question
        system_prompt, user_prompt = self._build_prompt(entity, summary, question, predictions)

        # 4. 调用 Claude API
        print(f"[Advisor] 调用 {self.model}...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        advice_text = response.content[0].text
        usage = response.usage

        # 5. 写入反馈存储（可选）
        session_id = None
        if self.feedback_store:
            session_id = self.feedback_store.log_advice(
                entity=entity,
                kg_summary=summary,
                advice_text=advice_text,
                model=self.model,
                time_window=n_recent_weeks,
            )
            print(f"[Advisor] 建议已记录（session #{session_id}），可用 advisor.feedback({session_id}, +1/-1/0) 评分")

        footer = (
            f"\n\n---\n"
            f"数据来源：FinDKG-full | 时间段：{summary.get('period')} | "
            f"事件数：{summary.get('total_events')} | "
            f"预测信号：{'有' if predictions else '无（未训练）'} | "
            f"Token用量：{usage.input_tokens} in / {usage.output_tokens} out"
        )

        return session_id, advice_text + footer

    # ── News Agent 接口（供 Orchestrator 使用）────────────────────

    def get_news_signal(
        self,
        entity: str,
        graph,
        n_recent_weeks: int = config.DEFAULT_WEEKS,
    ) -> dict:
        """
        News Agent 标准输出接口：返回结构化信号 dict，供 Orchestrator 消费。

        Returns
        -------
        {
            "entity":           str,
            "period":           str,
            "total_events":     int,
            "positive_impacts": list[dict],
            "negative_impacts": list[dict],
            "other_relations":  list[dict],
            "predictions":      list[dict] | None,
            "has_predictions":  bool,
            "kg_summary":       dict,       # 原始 KG 摘要
        }
        若实体在 KG 中无数据，返回 {"entity": entity, "error": str}
        """
        print(f"[NewsAgent] 查询知识图谱：{entity}（最近{n_recent_weeks}周）...")
        summary = graph.get_impact_summary(entity, n_recent_weeks=n_recent_weeks)

        if summary.get("total_events", 0) == 0:
            return {"entity": entity, "error": f"KG 中未找到 '{entity}' 的近期数据"}

        predictions = None
        if self.predictor and self.predictor.available:
            print(f"[NewsAgent] 运行 KGTransformer 预测...")
            predictions = self.predictor.predict_next(entity, graph, top_k=10)

        return {
            "entity":           entity,
            "period":           summary.get("period", "未知"),
            "total_events":     summary.get("total_events", 0),
            "positive_impacts": summary.get("positive_impacts", []),
            "negative_impacts": summary.get("negative_impacts", []),
            "other_relations":  summary.get("other_relations", []),
            "predictions":      predictions,
            "has_predictions":  predictions is not None,
            "kg_summary":       summary,
        }

    def feedback(self, session_id: int, rating: int, note: str = ""):
        """
        对某次建议评分（需要初始化时传入 feedback_store）。

        Parameters
        ----------
        session_id : advise() 返回的第一个值
        rating     : +1（好）/ 0（中性）/ -1（差）
        note       : 可选备注
        """
        if self.feedback_store is None:
            print("[Advisor] ⚠️  未配置 FeedbackStore，评分功能不可用。")
            return
        if session_id is None:
            print("[Advisor] ⚠️  session_id 为 None，无法评分。")
            return
        self.feedback_store.rate(session_id, rating, note)

    def advise_comparison(
        self,
        entities: list[str],
        graph,
        n_recent_weeks: int = config.DEFAULT_WEEKS,
    ) -> str:
        """对多个实体做对比分析"""
        summaries = {}
        for entity in entities:
            try:
                summaries[entity] = graph.get_impact_summary(entity, n_recent_weeks=n_recent_weeks)
            except ValueError as e:
                summaries[entity] = {"error": str(e)}

        context_parts = []
        for entity, summary in summaries.items():
            if "error" in summary:
                context_parts.append(f"【{entity}】查询失败：{summary['error']}")
                continue
            pos_count = len(summary.get("positive_impacts", []))
            neg_count = len(summary.get("negative_impacts", []))
            context_parts.append(
                f"【{entity}】时间段：{summary.get('period')} | "
                f"总事件：{summary.get('total_events')} | "
                f"正面信号类别：{pos_count} | 负面信号类别：{neg_count}"
            )
            if summary.get("positive_impacts"):
                top_pos = summary["positive_impacts"][0]
                context_parts.append(
                    f"  最强正面：{top_pos['subject']} --[{top_pos['relation']}]--> {top_pos['object']} (×{top_pos['count']})"
                )
            if summary.get("negative_impacts"):
                top_neg = summary["negative_impacts"][0]
                context_parts.append(
                    f"  最强负面：{top_neg['subject']} --[{top_neg['relation']}]--> {top_neg['object']} (×{top_neg['count']})"
                )

        kg_context = "\n".join(context_parts)
        entity_list = "、".join(entities)

        system_prompt = "你是一位专业的金融分析师助手，擅长基于知识图谱信号进行多资产对比分析。"
        user_prompt = f"""以下是多个金融实体在知识图谱中的近期信号摘要（最近{n_recent_weeks}周）：

{kg_context}

请对以上实体（{entity_list}）进行横向对比分析，给出：
1. 各实体的信号强弱排序
2. 综合配置建议（例如：超配/标配/低配/回避）
3. 主要风险点

建议结构清晰，每个实体单独一段分析。"""

        print(f"[Advisor] 对比分析：{entity_list}...")
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
    from feedback_store import FeedbackStore
    from kg_predictor import KGPredictor

    graph = FinDKGGraph()
    store = FeedbackStore()
    predictor = KGPredictor()  # 无 checkpoint 时自动禁用

    advisor = AssetAdvisor(
        predictor=predictor,
        feedback_store=store,
    )

    print("\n=== 单实体建议 ===")
    session_id, advice = advisor.advise("Apple Inc.", graph, n_recent_weeks=12)
    print(advice)

    # 模拟用户评分
    if session_id:
        advisor.feedback(session_id, rating=1, note="信号准确")
