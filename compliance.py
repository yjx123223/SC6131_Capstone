"""
compliance.py
--------------
ComplianceChecker：报告发布前的确定性合规检查 + 置信度兜底。

为什么不用 LLM 做合规（python-version 的 ComplianceAgent 是 LLM 打分）：
  合规规则需要可审计、可复现——同一份报告每次检查结果必须一致，
  并且每条问题都能指出触发它的具体规则。LLM 负责"判断质量"的部分
  已经由 Critic Agent 承担，这里只做规则能判定的事情。

在流水线中的位置（orchestrator.OrchestratorAgent.generate_report）：
  loop.run() → critic.review() → (可选) loop.revise()
  → ComplianceChecker.check()   ← 本模块：置信度封顶 + 合规问题清单
  → render_report()             ← 报告中展示合规结果
  → ensure_disclaimer()         ← 兜底确认免责声明存在

一、置信度兜底（把 Critic 的 confidence_adjustment 真正落实）
  1. Critic 要求 lower，而修订后的置信度没有低于原稿 → 在原稿基础上降一级
  2. 关键数据缺失时封顶：
     - 市场数据（行情/基本面）不可用 → 最高 low
     - 新闻 / SEC / 宏观中缺 2 项及以上 → 最高 low；缺 1 项 → 最高 medium
  3. Critic 要求 raise 不自动上调（保守原则），只记录一条 info

二、合规问题（severity: critical / warning / info）
  - critical：出现承诺收益、无风险、内幕信息等违规表述
  - warning ：风险提示过短；缺少关键信号；数据源失败但对应章节没有说明；
              新闻工具失败但情绪没标 unavailable；分析里没有注明数据日期；
              低置信度却给出强方向建议
              引用了图谱中不存在的事件编号；图谱发现的重大传导风险未被提及
  - info    ：置信度被下调 / 忽略上调请求等过程记录

  评分：100 - 30×critical - 10×warning；无 critical 且 ≥ 60 分视为合规。
"""

import re
from copy import deepcopy
from datetime import datetime
from typing import Optional

# ── 置信度 ──────────────────────────────────────────────────────
_CONF_ORDER = ["low", "medium", "high"]


def _rank(conf: Optional[str]) -> int:
    return _CONF_ORDER.index(conf) if conf in _CONF_ORDER else 0


def _lower_one(conf: Optional[str]) -> str:
    return _CONF_ORDER[max(_rank(conf) - 1, 0)]


# 数据工具 → 在报告里对应的章节字段
_DATA_TOOLS = {
    "query_market_data": ["fundamental_analysis", "technical_analysis"],
    "query_news":        ["news_sentiment_analysis"],
    "query_sec_filings": ["filings_analysis"],
    "query_macro":       ["macro_analysis"],
}
_SECONDARY_TOOLS = ("query_news", "query_sec_filings", "query_macro")

# 辅助数据工具：不参与置信度封顶，但缺失时对应章节同样需要说明
_AUX_TOOLS = {
    "query_company_graph": ["supply_chain_analysis"],
}
GRAPH_TOOL = "query_company_graph"
MATERIAL_RISK_SCORE = 0.7          # 图谱传导风险分数 ≥ 此值视为重大风险，报告必须提及
_EVENT_ID = re.compile(r"(?<![A-Za-z0-9])E(\d{1,4})(?![A-Za-z0-9])")

# 章节里用于说明"数据缺失"的关键词
_UNAVAILABLE_WORDS = ("不可用", "缺失", "无法获取", "未能获取", "获取失败", "暂无", "无数据", "unavailable", "not available")

# 违规表述（不区分大小写）
_PROHIBITED_PATTERNS = [
    (r"保证(收益|盈利|回报|上涨)", "承诺收益"),
    (r"稳赚|稳赚不赔|包赚|只赚不赔", "承诺收益"),
    (r"(零|无)风险(?!利率|收益率|资产)(投资|收益|套利)?", "宣称无风险"),   # 排除"无风险利率"等专业术语
    (r"必(涨|将上涨|然上涨)|一定(会)?上涨|肯定(会)?上涨", "确定性涨跌预测"),
    (r"内幕(消息|信息)", "暗示内幕信息"),
    (r"guarantee(d)?\s+(return|profit|gain)s?", "承诺收益"),
    (r"risk[-\s]?free(?!\s+(rate|asset))", "宣称无风险"),
    (r"insider\s+(information|tip)s?", "暗示内幕信息"),
]

_TEXT_FIELDS = [
    "executive_summary", "macro_analysis", "fundamental_analysis", "technical_analysis",
    "news_sentiment_analysis", "filings_analysis", "supply_chain_analysis",
    "recommendation_rationale", "risk_warnings",
]

_STRONG_RECOMMENDATIONS = {"增持", "减仓", "回避"}
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}\s*年")

_SEVERITY_PENALTY = {"critical": 30, "warning": 10, "info": 0}
PASS_SCORE = 60

DISCLAIMER_MARKER = "免责声明"
DISCLAIMER_TEMPLATE = """## 免责声明 / Disclaimer

本报告由 AI 系统基于公开数据自动生成，仅供研究与学习参考，不构成任何投资建议或要约。
报告中的数据来自第三方公开数据源，不保证其准确性、完整性与及时性；过往表现不代表未来收益。
投资有风险，任何投资决策应基于您自身的独立研究与风险承受能力。

This report is generated automatically from public data for informational purposes only and
does not constitute investment advice. Past performance is not indicative of future results.

报告生成时间：{timestamp}
"""


def _issue(severity: str, category: str, description: str, field: str = "") -> dict:
    return {"severity": severity, "category": category, "description": description, "field": field}


class ComplianceChecker:
    """
    使用示例
    --------
    >>> checker = ComplianceChecker()
    >>> result = checker.check(draft, critique, tool_log, original_confidence="high")
    >>> draft = result["draft"]          # 置信度已按规则封顶的新草稿（不修改入参）
    >>> result["is_compliant"], result["score"], result["issues"]
    """

    def check(
        self,
        draft: dict,
        critique: Optional[dict],
        tool_log: list,
        original_confidence: Optional[str] = None,
    ) -> dict:
        """
        Parameters
        ----------
        draft               : 修订后（若有）的草稿
        critique            : CriticAgent.review() 的结果
        tool_log            : OrchestratorLoop 的工具调用记录
        original_confidence : Critic 审查前原稿的置信度（用于判断修订是否已经下调过）

        Returns
        -------
        {
            "draft": dict,                      # 置信度封顶后的草稿副本
            "is_compliant": bool,
            "score": int,
            "issues": [ {severity, category, description, field}, ... ],
            "confidence": {"original": str, "final": str, "reasons": [str, ...]},
            "data_status": {tool_name: "ok" | "failed" | "not_called"},
        }
        """
        critique = critique or {}
        draft = deepcopy(draft)
        issues: list[dict] = []

        data_status = self._data_status(tool_log)

        conf_result = self._enforce_confidence(draft, critique, data_status, original_confidence)
        draft["confidence"] = conf_result["final"]
        if conf_result["final"] != conf_result["before"]:
            issues.append(_issue(
                "info", "confidence",
                f"置信度由 {conf_result['before']} 下调为 {conf_result['final']}：" + "；".join(conf_result["reasons"]),
                "confidence",
            ))
        if critique.get("confidence_adjustment") == "raise":
            issues.append(_issue("info", "confidence", "Critic 建议上调置信度，按保守原则未自动上调", "confidence"))

        issues += self._check_prohibited_language(draft)
        issues += self._check_risk_and_signals(draft)
        issues += self._check_data_disclosure(draft, data_status)
        issues += self._check_date_citation(draft, tool_log)
        issues += self._check_strength_vs_confidence(draft)
        issues += self._check_graph_citations(draft, tool_log)

        score = max(0, 100 - sum(_SEVERITY_PENALTY[i["severity"]] for i in issues))
        has_critical = any(i["severity"] == "critical" for i in issues)

        return {
            "draft": draft,
            "is_compliant": (not has_critical) and score >= PASS_SCORE,
            "score": score,
            "issues": issues,
            "confidence": {
                "original": original_confidence or conf_result["before"],
                "final": conf_result["final"],
                "reasons": conf_result["reasons"],
            },
            "data_status": data_status,
        }

    # ── 数据完整度 ──────────────────────────────────────────────

    @staticmethod
    def _data_status(tool_log: list) -> dict:
        """
        每个数据工具的状态：只要有一次成功调用就算 ok；
        调用过但全部失败算 failed；从未调用算 not_called。
        """
        status = {name: "not_called" for name in list(_DATA_TOOLS) + list(_AUX_TOOLS)}
        for entry in tool_log:
            name = entry.get("tool")
            if name not in status:
                continue
            ok = "error" not in (entry.get("result") or {})
            if ok:
                status[name] = "ok"
            elif status[name] != "ok":
                status[name] = "failed"
        return status

    @staticmethod
    def _enforce_confidence(draft: dict, critique: dict, data_status: dict, original: Optional[str]) -> dict:
        before = draft.get("confidence") if draft.get("confidence") in _CONF_ORDER else "low"
        final = before
        reasons = []

        # 1. 落实 Critic 的下调意见
        if critique.get("confidence_adjustment") == "lower":
            baseline = original if original in _CONF_ORDER else before
            if _rank(final) >= _rank(baseline):
                target = _lower_one(baseline)
                if _rank(target) < _rank(final):
                    final = target
                    reasons.append("Critic 要求下调置信度，修订稿未下调")

        # 2. 数据缺失封顶
        cap = "high"
        if data_status["query_market_data"] != "ok":
            cap = "low"
            reasons_cap = "市场数据（行情/基本面）不可用"
        else:
            missing = [t for t in _SECONDARY_TOOLS if data_status[t] != "ok"]
            if len(missing) >= 2:
                cap, reasons_cap = "low", f"{len(missing)} 项辅助数据不可用（{', '.join(missing)}）"
            elif len(missing) == 1:
                cap, reasons_cap = "medium", f"辅助数据不可用（{missing[0]}）"
            else:
                reasons_cap = ""
        if _rank(final) > _rank(cap):
            final = cap
            reasons.append(f"{reasons_cap}，置信度最高为 {cap}")

        return {"before": before, "final": final, "reasons": reasons}

    # ── 合规规则 ────────────────────────────────────────────────

    @staticmethod
    def _check_prohibited_language(draft: dict) -> list[dict]:
        issues = []
        texts = [(f, str(draft.get(f, ""))) for f in _TEXT_FIELDS]
        texts += [("key_signals", s) for s in draft.get("key_signals", [])]
        for field, text in texts:
            # 否定语境（"不保证收益"、"并非无风险"）不算违规
            for pattern, label in _PROHIBITED_PATTERNS:
                for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                    prefix = text[max(0, m.start() - 4):m.start()].lower()
                    if any(neg in prefix for neg in ("不", "非", "无法", "没有", "not ", "no ")):
                        continue
                    issues.append(_issue(
                        "critical", "regulatory",
                        f"出现违规表述「{m.group(0)}」（{label}）", field,
                    ))
        return issues

    @staticmethod
    def _check_risk_and_signals(draft: dict) -> list[dict]:
        issues = []
        if len(str(draft.get("risk_warnings", "")).strip()) < 10:
            issues.append(_issue("warning", "risk_disclosure", "风险提示缺失或过于简略（少于 10 字）", "risk_warnings"))
        if not draft.get("key_signals"):
            issues.append(_issue("warning", "data_citation", "未列出支撑结论的关键信号", "key_signals"))
        return issues

    @staticmethod
    def _check_data_disclosure(draft: dict, data_status: dict) -> list[dict]:
        """数据源失败 / 未调用时，对应章节必须说明数据缺失，避免"无数据却下结论" """
        issues = []
        for tool, fields in {**_DATA_TOOLS, **_AUX_TOOLS}.items():
            if data_status[tool] == "ok":
                continue
            for field in fields:
                text = str(draft.get(field, "")).lower()
                if not any(w.lower() in text for w in _UNAVAILABLE_WORDS):
                    state = "获取失败" if data_status[tool] == "failed" else "未查询"
                    issues.append(_issue(
                        "warning", "data_citation",
                        f"{tool} 数据{state}，但该章节未说明数据缺失", field,
                    ))
        if data_status["query_news"] != "ok" and draft.get("news_sentiment") != "unavailable":
            issues.append(_issue(
                "warning", "data_citation",
                f"新闻数据不可用，但新闻情绪标注为 {draft.get('news_sentiment')!r}（应为 unavailable）",
                "news_sentiment",
            ))
        return issues

    @staticmethod
    def _check_date_citation(draft: dict, tool_log: list) -> list[dict]:
        """市场数据可用时，基本面/技术面分析应注明数据日期，读者才能判断时效"""
        as_of = next(
            (e["result"].get("data_as_of") for e in tool_log
             if e.get("tool") == "query_market_data" and "error" not in (e.get("result") or {})),
            None,
        )
        if not as_of:
            return []
        text = f"{draft.get('technical_analysis', '')} {draft.get('fundamental_analysis', '')}"
        if as_of in text or _DATE_PATTERN.search(text):
            return []
        return [_issue(
            "warning", "data_citation",
            f"基本面/技术面分析未注明数据日期（行情截至 {as_of}）", "technical_analysis",
        )]

    @staticmethod
    def _check_strength_vs_confidence(draft: dict) -> list[dict]:
        if draft.get("recommendation") in _STRONG_RECOMMENDATIONS and draft.get("confidence") == "low":
            return [_issue(
                "warning", "bias",
                f"置信度为 low 却给出方向性较强的建议「{draft.get('recommendation')}」，建议改为持有/观望或说明理由",
                "recommendation",
            )]
        return []


    @staticmethod
    def _check_graph_citations(draft: dict, tool_log: list) -> list[dict]:
        """
        知识图谱引用检查：
          - cited_event_ids 或正文中出现的事件编号必须存在于图谱中（防止编造）
          - 图谱发现的重大传导风险（分数 ≥ MATERIAL_RISK_SCORE）必须在报告中被引用或提及
        """
        graph = next(
            (e["result"] for e in reversed(tool_log)
             if e.get("tool") == GRAPH_TOOL and "error" not in (e.get("result") or {})),
            None,
        )
        valid = set(graph.get("_event_ids", [])) if graph else set()

        cited = {str(x).strip().upper() for x in draft.get("cited_event_ids", []) or [] if str(x).strip()}
        body = " ".join(str(draft.get(f, "")) for f in _TEXT_FIELDS)
        body += " " + " ".join(draft.get("key_signals", []))
        mentioned = {f"E{m}" for m in _EVENT_ID.findall(body)}

        issues = []
        unknown = sorted((cited | mentioned) - valid, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        if unknown:
            reason = "图谱中不存在" if graph else "知识图谱不可用，却引用了"
            issues.append(_issue(
                "warning", "data_citation",
                f"{reason}事件编号：{', '.join(unknown)}", "cited_event_ids",
            ))

        if graph:
            referenced = cited | mentioned
            missed = sorted({
                r["event_id"] for r in graph.get("_risks", [])
                if r.get("score", 0) >= MATERIAL_RISK_SCORE and r["event_id"] not in referenced
            }, key=lambda x: int(x[1:]))
            if missed:
                issues.append(_issue(
                    "warning", "risk_disclosure",
                    f"知识图谱发现的重大传导风险未在报告中提及：{', '.join(missed)}",
                    "supply_chain_analysis",
                ))
        return issues


def disclaimer_block(timestamp: Optional[str] = None) -> str:
    return DISCLAIMER_TEMPLATE.format(timestamp=timestamp or datetime.now().strftime("%Y-%m-%d %H:%M"))


def ensure_disclaimer(report_md: str) -> tuple[str, bool]:
    """
    兜底：确认最终报告包含免责声明，缺失则追加。

    Returns (report_md, added)
    """
    if DISCLAIMER_MARKER in report_md:
        return report_md, False
    return report_md.rstrip("\n") + "\n\n---\n\n" + disclaimer_block(), True
