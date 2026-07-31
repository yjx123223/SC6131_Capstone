"""
tests/test_report_renderer.py
--------------------------------
report_renderer.render_report 是纯函数（拆分自 orchestrator.py 的
OrchestratorAgent._render_report），给定同样的 draft/critique/tool_log
输入，断言输出的 Markdown 包含预期的段落。
"""

from report_renderer import render_report


_DRAFT = {
    "executive_summary": "苹果近期正面信号占优，建议适度增持。",
    "macro_analysis": "利率见顶，宏观环境偏中性。",
    "entity_analysis": "近12周正面事件多于负面事件。",
    "recommendation": "增持",
    "recommendation_rationale": "正面信号强度高于负面信号。",
    "risk_warnings": "宏观不确定性仍存。",
    "confidence": "medium",
    "key_signals": ["Goldman Sachs Group 增持 Apple Inc.", "CPI 同比回落"],
}


def test_render_report_includes_all_sections():
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=[], model_name="claude-haiku-4-5")

    assert "投资建议报告：Apple Inc." in md
    assert "苹果近期正面信号占优" in md
    assert "**增持**" in md
    assert "中 ⚠️" in md  # confidence=medium 的展示文案
    assert "Goldman Sachs Group 增持 Apple Inc." in md
    assert "claude-haiku-4-5" in md


def test_render_report_no_critic_section_when_no_conflicts():
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=[], model_name="m")
    assert "Critic Agent 审查备注" not in md


def test_render_report_shows_critic_conflicts_when_present():
    critique = {
        "conflicts": ["宏观信号与个股信号方向相反"],
        "confidence_adjustment": "lower",
    }
    md = render_report("Apple Inc.", _DRAFT, critique=critique, tool_log=[], model_name="m")

    assert "Critic Agent 审查备注" in md
    assert "宏观信号与个股信号方向相反" in md
    assert "已下调" in md


def test_render_report_shows_tool_call_trajectory():
    tool_log = [
        {"tool": "query_kg_signals", "input": {"entity": "Apple Inc.", "weeks": 12}, "result": {}},
        {"tool": "query_macro", "input": {}, "result": {}},
    ]
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=tool_log, model_name="m")

    assert "query_kg_signals" in md
    assert "query_macro" in md
    assert "Tool Call Trajectory" in md


def test_render_report_empty_key_signals_shows_placeholder():
    draft = {**_DRAFT, "key_signals": []}
    md = render_report("Apple Inc.", draft, critique={}, tool_log=[], model_name="m")
    assert "（无）" in md
