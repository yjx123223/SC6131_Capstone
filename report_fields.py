"""
report_fields.py
-----------------
emit_report 草稿字段的唯一定义。

orchestrator_loop（emit_report schema / revise prompt）、critic（审查 prompt）
都通过这里读取字段列表和文字化方法，避免字段改名时多处漏改
（之前 entity_analysis 等字段名在三个文件里各写了一遍）。
"""

# (字段名, 中文标签)，顺序即 prompt 中的展示顺序
DRAFT_FIELDS = [
    ("executive_summary",        "执行摘要"),
    ("macro_analysis",           "宏观分析"),
    ("fundamental_analysis",     "基本面与估值分析"),
    ("technical_analysis",       "技术面分析"),
    ("news_sentiment_analysis",  "新闻舆情分析"),
    ("filings_analysis",         "监管申报分析"),
    ("supply_chain_analysis",    "产业链与竞争格局（知识图谱）"),
    ("recommendation",           "配置建议"),
    ("confidence",               "置信度"),
    ("recommendation_rationale", "建议理由"),
    ("risk_warnings",            "风险提示"),
]

RECOMMENDATION_ENUM = ["增持", "持有", "观望", "减仓", "回避"]
CONFIDENCE_ENUM = ["high", "medium", "low"]
SENTIMENT_ENUM = ["positive", "neutral", "negative", "unavailable"]


def format_draft(draft: dict) -> str:
    """把草稿 dict 文字化，供 Critic 审查 / 修订 prompt 使用"""
    lines = [f"- {label}：{draft.get(key, '')}" for key, label in DRAFT_FIELDS]
    lines.append(f"- 新闻情绪：{draft.get('news_sentiment', '')}")
    lines.append(f"- 关键信号：{'; '.join(draft.get('key_signals', []))}")
    lines.append(f"- 引用的图谱事件：{', '.join(draft.get('cited_event_ids', [])) or '无'}")
    return "\n".join(lines)
