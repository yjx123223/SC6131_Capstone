"""
tools/kg_live_tools.py
-----------------------
query_company_graph 工具：为目标公司现场构建一张实时知识图谱，
并基于图谱做供应链 / 竞争关系上的风险传导推理。

流程：
  1. 解析 ticker，读取人工种子关系，找出 1 跳邻居 + 2 跳上游供应商
  2. 目标公司的数据优先复用本轮 Orchestrator 已经拿到的结果
     （行情 → 行业节点；SEC → 申报节点；新闻 → 事件抽取），缺新闻时才自行拉取
  3. 并行拉取邻居公司的近期新闻
  4. 并行调用 EventExtractor 抽取事件，写入图谱（本体校验 + 去重）
  5. 风险传导推理（live_kg.queries，确定性规则）

返回值约定（在 tools 层 dict 约定基础上扩展）：
  - 普通字段：精简后的结果，会序列化发给模型
  - 以 "_" 开头的字段（完整图谱、Mermaid、带来源的完整风险列表）：
    只保留在 tool_log 中供报告渲染使用，OrchestratorLoop 发给模型前会剔除
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import config
from live_kg.event_extractor import EventExtractor
from live_kg.graph import CompanyGraph
from live_kg.queries import ROLE_ZH, find_neighbors, path_text, propagate_risks
from live_kg.seed import load_seed_relations
from . import news_tools
from .ticker_map import resolve_ticker

_MAX_EVENTS_TO_LLM = 8
_MAX_RISKS_TO_LLM = 8
_MAX_OPPS_TO_LLM = 5


def _compact_risk(r: dict) -> dict:
    return {k: r[k] for k in ("event_id", "neighbor", "hop", "role_zh", "impact", "score",
                              "polarity", "date", "summary", "path")}


def _compact_event(e: dict) -> dict:
    src = e["sources"][0] if e["sources"] else {}
    return {
        "id": e["id"], "event_type": e["event_type"], "polarity": e["polarity"],
        "date": e["date"], "summary": e["summary"], "confidence": e["confidence"],
        "source": f"{src.get('publisher', '')}: {src.get('title', '')}".strip(": "),
    }


def query_company_graph(
    entity: str,
    extractor: Optional[EventExtractor],
    prior_results: Optional[dict] = None,
    news_fetcher: Optional[Callable] = None,
    seed_rows: Optional[list[dict]] = None,
    max_hop1: Optional[int] = None,
    max_hop2: Optional[int] = None,
) -> dict:
    """
    Parameters
    ----------
    entity        : 公司名或 ticker
    extractor     : EventExtractor 实例（需要 anthropic client）
    prior_results : 本轮已获取的目标公司数据 {"query_market_data": {...}, "query_news": {...},
                    "query_sec_filings": {...}}，只会使用 ticker 与目标一致且没有 error 的结果
    news_fetcher  : (ticker, max_items) -> query_news 格式的结果；默认 news_tools.query_news
    seed_rows     : 种子关系（测试注入用）；默认读取 config.KG_SEED_PATH
    """
    resolved = resolve_ticker(entity)
    if "error" in resolved:
        return resolved
    if extractor is None:
        return {"error": "未配置事件抽取器（缺少 Anthropic client），无法构建知识图谱"}

    target = resolved["ticker"]
    fetch_news = news_fetcher or (lambda t, n: news_tools.query_news(t, max_items=n))
    warnings: list[str] = []

    prior = {
        name: r for name, r in (prior_results or {}).items()
        if isinstance(r, dict) and "error" not in r and str(r.get("ticker", "")).upper() == target
    }

    # 1. 种子关系 + 邻居
    if seed_rows is None:
        seed_rows, seed_errors = load_seed_relations()
        warnings += [f"种子文件：{e}" for e in seed_errors]
    graph = CompanyGraph()
    graph.load_seed(seed_rows)

    market = prior.get("query_market_data", {})
    company_name = (market.get("company") or {}).get("name")
    graph.add_company(target, name=company_name, is_target=True)

    neighbors = find_neighbors(graph, target, max_hop1, max_hop2)
    if not neighbors:
        warnings.append(f"种子关系中没有 {target} 的供应链/竞争/合作关系，图谱只包含其自身事件")
    tickers = [target] + [n["ticker"] for n in neighbors]
    graph.prune_companies(tickers)

    # 2. 目标公司的结构化数据（复用本轮结果）
    industry = (market.get("company") or {}).get("industry")
    if industry:
        graph.add_relation("IN_INDUSTRY", f"company:{target}", graph.add_industry(industry))
    for f in prior.get("query_sec_filings", {}).get("filings", []):
        fid = graph.add_filing(f.get("form", ""), f.get("filing_date", ""), f.get("url", ""))
        graph.add_relation("FILED", f"company:{target}", fid)

    # 3. 新闻：目标公司优先复用，邻居并行拉取
    news_by_ticker: dict[str, list] = {}
    news_failed: list[str] = []
    if "query_news" in prior:
        news_by_ticker[target] = prior["query_news"].get("articles", [])
    to_fetch = [t for t in tickers if t not in news_by_ticker]

    def _fetch(t):
        n = config.NEWS_MAX_ITEMS if t == target else config.KG_NEWS_PER_COMPANY
        try:
            return t, fetch_news(t, n)
        except Exception as e:
            return t, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=config.KG_MAX_WORKERS) as pool:
        for t, res in pool.map(_fetch, to_fetch):
            if "error" in res:
                news_failed.append(t)
            else:
                news_by_ticker[t] = res.get("articles", [])
    if news_failed:
        warnings.append(f"以下公司的新闻不可用：{', '.join(news_failed)}")

    # 4. 并行事件抽取
    known = {
        d["ticker"]: d.get("name") or d["ticker"]
        for _, d in graph.g.nodes(data=True) if d["type"] == "Company"
    }
    jobs = [(t, arts) for t, arts in news_by_ticker.items() if arts]

    def _extract(job):
        t, arts = job
        try:
            return t, arts, extractor.extract(t, arts, known)
        except Exception as e:
            return t, arts, {"events": [], "dropped": {}, "error": str(e)}

    extraction = {"companies_ok": [], "companies_failed": [], "events_extracted": 0, "dropped": {}}
    with ThreadPoolExecutor(max_workers=config.KG_MAX_WORKERS) as pool:
        results = list(pool.map(_extract, jobs))

    for t, arts, res in sorted(results, key=lambda x: tickers.index(x[0])):
        if res.get("error"):
            extraction["companies_failed"].append(t)
            warnings.append(res["error"])
        else:
            extraction["companies_ok"].append(t)
        for reason, cnt in res.get("dropped", {}).items():
            extraction["dropped"][reason] = extraction["dropped"].get(reason, 0) + cnt
        for ev in res.get("events", []):
            art = arts[ev["article_index"]]
            nid = graph.add_news(art.get("title", ""), art.get("publisher", ""),
                                 art.get("published_at", ""), art.get("url", ""))
            if graph.add_event(ev["event_type"], ev["polarity"], ev["date"], ev["summary"],
                               ev["tickers"], [nid], ev["confidence"]):
                extraction["events_extracted"] += 1

    # 5. 推理
    reasoning = propagate_risks(graph, target, neighbors)
    target_events = graph.events_for(target)
    highlight = {r["event_id"] for r in reasoning["risks"][:_MAX_RISKS_TO_LLM]} | \
                {e["id"] for e in target_events[:_MAX_EVENTS_TO_LLM]}

    return {
        "ticker":  target,
        "entity":  resolved["entity"],
        "source":  "实时知识图谱（人工种子关系 + Yahoo Finance 新闻 + LLM 事件抽取）",
        "neighbors": [
            {"ticker": n["ticker"], "hop": n["hop"],
             "roles": [ROLE_ZH[r] for r in n["roles"]], "via": n["via"], "path": path_text(n["path"])}
            for n in neighbors
        ],
        "target_events":    [_compact_event(e) for e in target_events[:_MAX_EVENTS_TO_LLM]],
        "propagated_risks": [_compact_risk(r) for r in reasoning["risks"][:_MAX_RISKS_TO_LLM]],
        "opportunities":    [_compact_risk(r) for r in reasoning["opportunities"][:_MAX_OPPS_TO_LLM]],
        "stats": {**graph.stats(), "extraction": extraction, "news_failed": news_failed},
        "warnings": warnings,
        # ── 以下字段不发给模型，仅供报告渲染 / 存档 ──
        "_graph":         graph.to_dict(),
        "_mermaid":       graph.to_mermaid(config.KG_MAX_EVENTS_PER_COMPANY_IN_DIAGRAM, highlight),
        "_risks":         reasoning["risks"],
        "_opportunities": reasoning["opportunities"],
        "_target_events": target_events,
        "_event_ids":     sorted(graph.event_ids(), key=lambda x: int(x[1:])),
    }
