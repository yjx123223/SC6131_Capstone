"""
tests/test_report_renderer.py
--------------------------------
report_renderer.render_report 是纯函数，给定同样的 draft/critique/tool_log
输入，断言输出的 Markdown 包含预期的段落。
"""

from report_renderer import render_report, render_data_sources


_DRAFT = {
    "executive_summary": "苹果近期正面信号占优，建议适度增持。",
    "macro_analysis": "利率见顶，宏观环境偏中性。",
    "fundamental_analysis": "PE 33 倍，利润率 24%。",
    "technical_analysis": "股价站上 MA20，RSI 61。",
    "news_sentiment_analysis": "近两周新品发布新闻偏正面。",
    "news_sentiment": "positive",
    "filings_analysis": "7 月底提交 10-Q。",
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
    # 备注行与分隔线之间必须有空行，否则 Markdown 会把它渲染成标题
    assert "*置信度调整：已下调*\n\n---" in md


def test_render_report_shows_tool_call_trajectory():
    tool_log = [
        {"tool": "query_market_data", "input": {"entity": "Apple Inc.", "period": "3mo"}, "result": {}},
        {"tool": "query_macro", "input": {}, "result": {}},
    ]
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=tool_log, model_name="m")

    assert "query_market_data" in md
    assert "query_macro" in md
    assert "Tool Call Trajectory" in md


def test_render_report_empty_key_signals_shows_placeholder():
    draft = {**_DRAFT, "key_signals": []}
    md = render_report("Apple Inc.", draft, critique={}, tool_log=[], model_name="m")
    assert "（无）" in md


def test_render_report_includes_new_analysis_sections():
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=[], model_name="m")
    assert "## 基本面与估值" in md and "PE 33 倍" in md
    assert "## 技术面" in md and "RSI 61" in md
    assert "## 新闻舆情（整体情绪：偏正面）" in md
    assert "## 监管申报（SEC EDGAR）" in md and "10-Q" in md
    assert "（本次未调用任何数据工具）" in md


def test_render_data_sources_is_built_from_tool_log():
    tool_log = [
        {"tool": "query_market_data", "input": {}, "result": {
            "source": "Yahoo Finance (yfinance)", "ticker": "AAPL", "data_as_of": "2026-09-15",
            "period": "3mo", "warnings": ["公司信息/财务指标为空"]}},
        {"tool": "query_news", "input": {}, "result": {
            "source": "Yahoo Finance News (yfinance)", "lookback_days": 14, "article_count": 1,
            "articles": [{"title": "Apple launches iPhone", "publisher": "Reuters",
                          "published_at": "2026-09-10T10:00+00:00", "url": "https://n.example/1"}]}},
        {"tool": "query_sec_filings", "input": {}, "result": {
            "source": "SEC EDGAR", "company": "Apple Inc.", "cik": 320193,
            "filings": [{"form": "10-Q", "filing_date": "2026-07-31", "url": "https://sec.example/q"}]}},
        {"tool": "query_macro", "input": {}, "result": {"indicators": {
            "vix": {"value": 15.0, "date": "2026-09-15"},
            "cpi_yoy": {"value": 2.9, "date": "2026-08-01"},
            "unemployment": {"value": None, "date": "N/A"}}}},
        {"tool": "query_news", "input": {}, "result": {"error": "新闻源不可用"}},
    ]
    md = render_data_sources(tool_log)

    assert "行情截至 **2026-09-15**" in md
    assert "⚠️ 公司信息/财务指标为空" in md
    assert "[Apple launches iPhone](https://n.example/1)" in md
    assert "[10-Q 2026-07-31](https://sec.example/q)" in md
    assert "指标日期 2026-08-01 ~ 2026-09-15" in md
    assert "❌ `query_news` 数据不可用：新闻源不可用" in md


def test_render_report_unknown_sentiment_placeholder():
    draft = {**_DRAFT}
    draft.pop("news_sentiment")
    md = render_report("Apple Inc.", draft, critique={}, tool_log=[], model_name="m")
    assert "整体情绪：未标注" in md


def test_render_report_always_contains_disclaimer():
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=[], model_name="m")
    assert "## 免责声明 / Disclaimer" in md


def test_render_report_without_compliance_has_no_compliance_section():
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=[], model_name="m")
    assert "## 合规检查" not in md


def test_render_report_shows_compliance_and_confidence_downgrade():
    compliance = {
        "is_compliant": False,
        "score": 70,
        "issues": [{"severity": "critical", "category": "regulatory",
                    "description": "出现违规表述「稳赚」| 测试", "field": "risk_warnings"}],
        "confidence": {"original": "high", "final": "medium", "reasons": ["x"]},
    }
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=[], model_name="m", compliance=compliance)

    assert "## 合规检查" in md
    assert "⚠️ 未通过" in md and "70/100" in md
    assert "🔴 critical | regulatory | risk_warnings |" in md
    assert "稳赚」\\| 测试" in md                  # 表格内的 | 需转义
    assert "置信度：中 ⚠️（原为高，已按审查/合规规则下调）" in md


def test_render_compliance_without_issues():
    from report_renderer import render_compliance
    md = render_compliance({"is_compliant": True, "score": 100, "issues": []})
    assert "✅ 通过" in md and "未发现问题" in md


# ── 知识图谱段落 ────────────────────────────────────────────────

_GRAPH_RESULT = {
    "ticker": "AAPL",
    "source": "实时知识图谱",
    "neighbors": [
        {"ticker": "TSM", "hop": 1, "roles": ["供应商"], "via": None, "path": "TSM ─SUPPLIES_TO→ AAPL"},
        {"ticker": "ASML", "hop": 2, "roles": ["上游供应商"], "via": "TSM",
         "path": "ASML ─SUPPLIES_TO→ TSM ─SUPPLIES_TO→ AAPL"},
    ],
    "stats": {"node_count": 9, "edge_count": 11, "rejected_count": 0,
              "extraction": {"companies_ok": ["AAPL", "TSM"], "companies_failed": ["ASML"],
                             "events_extracted": 2, "dropped": {"公司无法链接": 2}}},
    "warnings": ["抽取失败 ASML"],
    "_mermaid": "graph LR\n  n0[\"AAPL\"]",
    "_risks": [{
        "event_id": "E2", "neighbor": "TSM", "hop": 1, "role_zh": "供应商", "impact": "供应风险",
        "score": 0.8, "event_type": "SUPPLY_CHAIN", "date": "2026-09-12", "summary": "产能|受限",
        "path": "TSM ─SUPPLIES_TO→ AAPL",
        "sources": [{"title": "TSMC capacity", "url": "https://n/2", "publisher": "Bloomberg"}],
    }],
    "_opportunities": [{"event_id": "E3", "neighbor": "MSFT", "date": "2026-09-11", "summary": "对手承压"}],
    "_target_events": [{
        "id": "E1", "event_type": "PRODUCT", "polarity": "positive", "date": "2026-09-10",
        "summary": "iPhone 18 发布", "confidence": 0.9,
        "sources": [{"title": "Apple unveils", "url": "", "publisher": "Reuters"}],
    }],
}


def test_render_knowledge_graph_full():
    from report_renderer import render_knowledge_graph
    md = render_knowledge_graph(_GRAPH_RESULT, cited_event_ids=["e2"])

    assert "```mermaid\ngraph LR" in md
    assert "| ASML | 2 | 上游供应商 | ASML ─SUPPLIES_TO→ TSM ─SUPPLIES_TO→ AAPL |" in md
    assert "| ★E2 | 供应风险 | 0.80 | TSM（供应商） | 2026-09-12 | 供应链：产能\\|受限 |" in md
    assert "[TSMC capacity](https://n/2)" in md
    assert "- E3 MSFT（2026-09-11）：对手承压" in md
    assert "| E1 | 2026-09-10 | 产品 | 正面 | iPhone 18 发布 | Apple unveils |" in md
    assert "失败 1 家" in md and "公司无法链接 2" in md
    assert "人工整理的种子数据" in md
    assert md.startswith("<details open>") and md.rstrip().endswith("</details>")


def test_render_knowledge_graph_unavailable_and_empty():
    from report_renderer import render_knowledge_graph
    assert "知识图谱不可用" in render_knowledge_graph(None)
    md = render_knowledge_graph({"stats": {}, "neighbors": []})
    assert "未发现相关公司的风险传导事件" in md
    assert "```mermaid" not in md


def test_render_report_includes_supply_chain_and_graph_sections():
    draft = {**_DRAFT, "supply_chain_analysis": "供应商台积电产能受限（E2）。", "cited_event_ids": ["E2"]}
    tool_log = [{"tool": "query_company_graph", "input": {"entity": "AAPL"}, "result": _GRAPH_RESULT}]
    md = render_report("Apple Inc.", draft, critique={}, tool_log=tool_log, model_name="m")

    assert "## 产业链与竞争格局（知识图谱）\n\n供应商台积电产能受限（E2）。" in md
    assert "★E2" in md
    assert "- 知识图谱：实时知识图谱，9 个节点 / 11 条边，抽取事件 2 个" in md
    assert "  - ⚠️ 抽取失败 ASML" in md
    assert "实时知识图谱 + Critic reflection" in md
    # 图谱段落位于"综合配置建议"之前
    assert md.index("## 产业链与竞争格局") < md.index("```mermaid") < md.index("## 综合配置建议")


def test_render_report_graph_failed():
    tool_log = [{"tool": "query_company_graph", "input": {}, "result": {"error": "boom"}}]
    md = render_report("Apple Inc.", _DRAFT, critique={}, tool_log=tool_log, model_name="m")
    assert "知识图谱不可用（本次未成功构建）" in md
    assert "❌ `query_company_graph` 数据不可用：boom" in md
