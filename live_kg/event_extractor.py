"""
live_kg/event_extractor.py
---------------------------
EventExtractor：用 LLM 从某家公司的近期新闻中抽取结构化事件。

设计要点：
  - 用 tool schema + tool_choice 强制结构化输出，事件类型 / 正负面都是本体里的枚举
  - 提示词里给出图中已有公司及其 ticker，要求优先输出 ticker，降低实体链接难度
  - 抽取结果逐条校验（事后不信任 LLM）：
      article_index 越界       → 丢弃（证据必须真实存在）
      polarity 不合法          → 丢弃
      event_type 不在枚举内    → 归为 OTHER（留作本体迭代的线索）
      公司名无法链接到图中公司 → 去掉该公司；一个公司都不剩则丢弃事件
      未给出受影响公司         → 默认是新闻所属的公司
  - 失败（API 异常 / 输出被截断）返回 {"events": [], "error": ...}，不抛异常，
    由调用方决定是否影响整体流程

每家公司调用一次（由 tools/kg_live_tools.py 并行调度）。
"""

from typing import Iterable, Optional

import config
from tools.ticker_map import resolve_ticker
from . import ontology as ont

EXTRACT_TOOL = {
    "name": "record_events",
    "description": "记录从新闻中抽取出的公司相关事件。没有值得记录的事件时传空数组。",
    "input_schema": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_index": {
                            "type": "integer",
                            "description": "事件依据的新闻编号（方括号里的数字）",
                        },
                        "event_type": {
                            "type": "string",
                            "enum": list(ont.EVENT_TYPES),
                            "description": "事件类型",
                        },
                        "polarity": {
                            "type": "string",
                            "enum": list(ont.POLARITIES),
                            "description": "该事件对受影响公司是正面、负面还是中性",
                        },
                        "affected_companies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "受该事件直接影响的公司，优先使用给定列表中的 ticker",
                        },
                        "summary": {
                            "type": "string",
                            "description": "中文一句话概括事件，不超过 60 字",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0~1：事件是否确实发生、对公司影响是否明确",
                        },
                    },
                    "required": ["article_index", "event_type", "polarity", "affected_companies", "summary", "confidence"],
                },
            },
        },
        "required": ["events"],
    },
}

SYSTEM_PROMPT = """你是金融信息抽取器，负责从新闻中抽取对公司有实质影响的事件。

规则：
- 只依据给出的新闻内容，不要推测或补充新闻中没有的信息
- 每条新闻最多抽取 2 个事件；纯行情播报、泛泛的市场评论、与公司无实质关系的新闻可以跳过
- 同一事件被多条新闻报道时，只记录一次，使用最具代表性的那条新闻编号
- polarity 是事件对 affected_companies 的影响方向
- affected_companies 优先使用"已知公司"列表中的 ticker；列表外的公司写英文正式名称
- summary 用中文，不超过 60 字"""

_TYPE_ZH = "、".join(f"{k}（{v}）" for k, v in ont.EVENT_TYPES.items())


def _format_articles(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles):
        date = str(a.get("published_at", ""))[:10]
        lines.append(f"[{i}] {date} | {a.get('publisher', '')} | {a.get('title', '')}")
        if a.get("summary"):
            lines.append(f"    {a['summary']}")
    return "\n".join(lines)


class EventExtractor:
    """
    使用示例
    --------
    >>> extractor = EventExtractor(anthropic_client)
    >>> result = extractor.extract("TSM", articles, known_companies={"TSM": "TSMC", "AAPL": "Apple"})
    >>> result["events"]   # [{"article_index", "event_type", "polarity", "tickers", "summary", "confidence", "date"}]
    """

    def __init__(self, client, model: str = config.KG_EXTRACT_MODEL,
                 max_tokens: int = config.KG_EXTRACT_MAX_TOKENS):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, focus_ticker: str, articles: list[dict],
                known_companies: Optional[dict[str, str]] = None) -> dict:
        """
        Parameters
        ----------
        focus_ticker    : 这批新闻所属的公司
        articles        : news_tools.query_news 返回的 articles
        known_companies : 图中已有公司 {ticker: 名称}，只有能链接到这些 ticker 的公司才会保留

        Returns
        -------
        {"events": [...], "dropped": {原因: 数量}, "error": str（仅失败时）}
        """
        focus_ticker = focus_ticker.upper()
        known = {t.upper(): n for t, n in (known_companies or {focus_ticker: focus_ticker}).items()}
        known.setdefault(focus_ticker, focus_ticker)
        dropped: dict[str, int] = {}

        if not articles:
            return {"events": [], "dropped": dropped}

        company_list = "、".join(f"{t}（{n}）" for t, n in sorted(known.items()))
        user_prompt = (
            f"新闻所属公司：{focus_ticker}（{known[focus_ticker]}）\n"
            f"已知公司：{company_list}\n"
            f"可选事件类型：{_TYPE_ZH}\n\n"
            f"新闻列表：\n{_format_articles(articles)}\n\n"
            "请调用 record_events 记录事件。"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[EXTRACT_TOOL],
                tool_choice={"type": "tool", "name": "record_events"},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            return {"events": [], "dropped": dropped, "error": f"事件抽取调用失败（{focus_ticker}）：{e}"}

        if getattr(response, "stop_reason", None) == "max_tokens":
            return {"events": [], "dropped": dropped,
                    "error": f"事件抽取输出被截断（{focus_ticker}），请调大 KG_EXTRACT_MAX_TOKENS"}

        raw_events = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_events":
                raw_events = (block.input or {}).get("events")
                break
        if not isinstance(raw_events, list):
            return {"events": [], "dropped": dropped, "error": f"事件抽取未返回结构化结果（{focus_ticker}）"}

        events = []
        for raw in raw_events:
            ev = self._validate(raw, focus_ticker, articles, known, dropped)
            if ev is not None:
                events.append(ev)
        return {"events": events, "dropped": dropped}

    @staticmethod
    def _bump(dropped: dict, reason: str, n: int = 1):
        dropped[reason] = dropped.get(reason, 0) + n

    def _validate(self, raw, focus_ticker, articles, known, dropped) -> Optional[dict]:
        if not isinstance(raw, dict):
            self._bump(dropped, "格式错误")
            return None

        idx = raw.get("article_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(articles):
            self._bump(dropped, "新闻编号无效")
            return None

        polarity = raw.get("polarity")
        if polarity not in ont.POLARITIES:
            self._bump(dropped, "正负面无效")
            return None

        event_type = raw.get("event_type")
        if event_type not in ont.EVENT_TYPES:
            self._bump(dropped, "事件类型归为 OTHER")
            event_type = "OTHER"

        tickers = self._link_companies(raw.get("affected_companies") or [], known, dropped)
        if not raw.get("affected_companies"):
            tickers = [focus_ticker]
        if not tickers:
            self._bump(dropped, "无可链接的公司")
            return None

        summary = str(raw.get("summary") or articles[idx].get("title", "")).strip()[:80]
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        return {
            "article_index": idx,
            "event_type":    event_type,
            "polarity":      polarity,
            "tickers":       tickers,
            "summary":       summary,
            "confidence":    confidence,
            "date":          str(articles[idx].get("published_at", ""))[:10],
        }

    def _link_companies(self, names: Iterable, known: dict, dropped: dict) -> list[str]:
        """实体链接：公司名 → ticker，且必须是图中已有公司"""
        tickers = []
        for name in names:
            if not isinstance(name, str) or not name.strip():
                continue
            candidate = name.strip()
            upper = candidate.upper()
            if upper in known:
                ticker = upper
            else:
                resolved = resolve_ticker(candidate)
                ticker = resolved.get("ticker")
            if ticker is None or ticker not in known:
                self._bump(dropped, "公司无法链接")
                continue
            if ticker not in tickers:
                tickers.append(ticker)
        return tickers
