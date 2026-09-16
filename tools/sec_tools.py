"""
tools/sec_tools.py
-------------------
query_sec_filings 工具的实现：从 SEC EDGAR 获取目标公司最近的监管申报。

数据接口（均免费，无需 API Key，但必须带声明身份的 User-Agent）：
  - https://www.sec.gov/files/company_tickers.json      ticker → CIK
  - https://data.sec.gov/submissions/CIK##########.json  最近申报列表

与 python-version 的差异：
  - 失败 / 无近期申报 → 返回 {"error": ...}，不降级为 mock 申报
  - 只保留最近 config.SEC_LOOKBACK_DAYS 天内的申报
  - 文档链接用 accession number + primaryDocument 拼出真实 URL
  - User-Agent 未配置时直接报错（SEC 会拒绝匿名请求）

测试方式：通过 http_get 参数注入替身，不联网。
"""

from datetime import date, datetime, timedelta
from typing import Callable, Optional

import config
from .ticker_map import resolve_ticker
from .yf_client import to_utc, utc_now

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# ticker → (cik, title) 的进程内缓存；company_tickers.json 约 1MB，没必要每次都拉
_cik_cache: dict[str, tuple[int, str]] = {}


def _default_http_get(url: str, user_agent: str) -> dict:
    import httpx

    resp = httpx.get(
        url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=config.HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def _lookup_cik(ticker: str, http_get: Callable, user_agent: str) -> Optional[tuple[int, str]]:
    key = ticker.upper()
    if not _cik_cache:
        data = http_get(_TICKERS_URL, user_agent)
        for entry in data.values():
            t = str(entry.get("ticker", "")).upper()
            if t:
                _cik_cache[t] = (int(entry["cik_str"]), entry.get("title", ""))
    # SEC 用 "BRK-B"，部分数据源用 "BRK.B"，两种写法都试
    return _cik_cache.get(key) or _cik_cache.get(key.replace(".", "-"))


def clear_cache():
    """测试用：清空 CIK 缓存"""
    _cik_cache.clear()


def query_sec_filings(
    entity: str,
    form_types: Optional[list[str]] = None,
    max_items: int = config.SEC_MAX_FILINGS,
    lookback_days: int = config.SEC_LOOKBACK_DAYS,
    user_agent: Optional[str] = None,
    http_get: Optional[Callable] = None,
    now: Optional[datetime] = None,
) -> dict:
    """
    获取目标公司最近的 SEC 申报（10-K / 10-Q / 8-K 等）。

    Returns
    -------
    成功：
        {
            "ticker", "cik", "company", "source", "lookback_days",
            "filings": [{"form", "filing_date", "report_date", "description", "url"}, ...],
        }
    失败：{"error": str}
    """
    ua = config.get_sec_user_agent(user_agent)
    if not ua:
        return {
            "error": (
                "未配置 SEC_EDGAR_USER_AGENT，SEC 会拒绝匿名请求。"
                "请设置：export SEC_EDGAR_USER_AGENT='Your Name your.email@example.com'"
            )
        }

    resolved = resolve_ticker(entity)
    if "error" in resolved:
        return resolved
    ticker = resolved["ticker"]

    http_get = http_get or _default_http_get
    today: date = (to_utc(now) if now is not None else utc_now()).date()
    forms = {f.upper() for f in (form_types or config.SEC_DEFAULT_FORMS)}
    max_items = max(1, min(int(max_items), 20))

    try:
        found = _lookup_cik(ticker, http_get, ua)
    except Exception as e:
        return {"error": f"SEC CIK 查询失败：{e}"}
    if not found:
        return {"error": f"SEC EDGAR 中未找到 {ticker} 对应的公司（可能不是美国上市公司）"}
    cik, title = found

    try:
        data = http_get(_SUBMISSIONS_URL.format(cik=cik), ua)
    except Exception as e:
        return {"error": f"SEC 申报列表拉取失败（CIK {cik}）：{e}"}

    recent = (data.get("filings") or {}).get("recent") or {}
    cols = ["form", "filingDate", "reportDate", "accessionNumber", "primaryDocument", "primaryDocDescription"]
    columns = {c: recent.get(c) or [] for c in cols}
    n = len(columns["form"])
    cutoff = today - timedelta(days=lookback_days)

    def _col(name, i):
        vals = columns[name]
        return vals[i] if i < len(vals) else ""

    filings = []
    for i in range(n):
        form = str(_col("form", i)).upper()
        if form not in forms:
            continue
        try:
            filed = date.fromisoformat(_col("filingDate", i))
        except (TypeError, ValueError):
            continue
        if filed < cutoff:
            continue
        acc = str(_col("accessionNumber", i))
        doc = _col("primaryDocument", i)
        url = (
            _ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=doc)
            if acc and doc else ""
        )
        filings.append({
            "form":        form,
            "filing_date": filed.isoformat(),
            "report_date": _col("reportDate", i) or None,
            "description": _col("primaryDocDescription", i) or form,
            "url":         url,
        })

    if not filings:
        return {"error": f"{ticker} 最近 {lookback_days} 天内没有 {', '.join(sorted(forms))} 类申报"}

    filings.sort(key=lambda f: f["filing_date"], reverse=True)

    return {
        "ticker":        ticker,
        "cik":           cik,
        "company":       data.get("name") or title,
        "source":        "SEC EDGAR",
        "lookback_days": lookback_days,
        "filings":       filings[:max_items],
    }
