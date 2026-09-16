"""
tools/ticker_map.py
--------------------
实体名 → 股票代码（ticker）解析。

背景：Orchestrator 收到的是 "Apple Inc." 这类公司名（沿用 FinDKG 的实体
命名习惯），而 yfinance / SEC EDGAR 需要的是 "AAPL" 这类 ticker。

解析策略（按顺序）：
  1. 规范化后查内置映射表（大小写、标点、Inc./Corp. 等后缀不敏感）
  2. 输入本身看起来就是 ticker（全大写字母，1~5 位，可带 . 或 -，
     如 "AAPL"、"BRK-B"）→ 直接使用
  3. 都不满足 → 返回 {"error": ...}，不做模糊猜测
     （猜错 ticker 会拿到另一家公司的数据，比报错更危险）

扩展方式：往 _ENTITY_TO_TICKER 里加条目即可。
"""

import re

# 规范化后的公司名 → ticker（key 必须是 _normalize() 之后的形式）
_ENTITY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "amazoncom": "AMZN",
    "meta platforms": "META",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "intel": "INTC",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "ibm": "IBM",
    "international business machines": "IBM",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "taiwan semiconductor manufacturing": "TSM",
    "taiwan semiconductor": "TSM",
    "tsmc": "TSM",
    "asml": "ASML",
    "amazon web services": "AMZN",
    "aws": "AMZN",
    "ford": "F",
    "goldman sachs": "GS",
    "jpmorgan chase": "JPM",
    "jpmorgan": "JPM",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "citigroup": "C",
    "wells fargo": "WFC",
    "berkshire hathaway": "BRK-B",
    "blackrock": "BLK",
    "visa": "V",
    "mastercard": "MA",
    "walmart": "WMT",
    "costco": "COST",
    "coca-cola": "KO",
    "cocacola": "KO",
    "pepsico": "PEP",
    "mcdonalds": "MCD",
    "nike": "NKE",
    "walt disney": "DIS",
    "disney": "DIS",
    "boeing": "BA",
    "exxon mobil": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "johnson johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "general motors": "GM",
    "ford motor": "F",
    "alibaba": "BABA",
    "uber technologies": "UBER",
    "uber": "UBER",
}

# 规范化时去掉的公司后缀（只去掉末尾的，避免误删名称中间的词）
_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "plc", "group", "holding", "holdings", "the", "nv", "sa", "ag",
}

_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def _normalize(name: str) -> str:
    """小写、去标点、去掉开头的 the 和末尾的公司后缀"""
    s = name.lower().replace("&", " ").replace("'", "").replace(".", "")
    s = re.sub(r"[^\w\s\-]", " ", s)
    words = s.split()
    if words and words[0] == "the":
        words = words[1:]
    while words and words[-1] in _SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def resolve_ticker(entity: str) -> dict:
    """
    把实体名或 ticker 解析成 ticker。

    Returns
    -------
    成功：{"ticker": str, "entity": str, "matched_by": "mapping" | "ticker"}
    失败：{"error": str}
    """
    raw = (entity or "").strip()
    if not raw:
        return {"error": "实体名为空"}

    key = _normalize(raw)
    if key in _ENTITY_TO_TICKER:
        return {"ticker": _ENTITY_TO_TICKER[key], "entity": raw, "matched_by": "mapping"}

    if _TICKER_PATTERN.match(raw):
        return {"ticker": raw, "entity": raw, "matched_by": "ticker"}

    return {
        "error": (
            f"无法将 '{raw}' 解析为股票代码：不在内置映射表中。"
            "请直接传入 ticker（如 AAPL），或在 tools/ticker_map.py 中补充映射。"
        )
    }
