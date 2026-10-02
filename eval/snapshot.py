"""
eval/snapshot.py
-----------------
工具返回结果的录制与回放。

为什么需要：三组实验（A/B/C）必须看到**完全一样的数据**，否则新闻一变就没法比较。
录制一次，之后所有实验都从快照回放，同时也避免反复请求外部接口。

录制范围（覆盖三组可能用到的全部调用）：
  - universe 中每家公司的 query_market_data 与 query_news
    （B 组允许查询任意相关公司，所以邻居也要录）
  - 每家目标公司的 query_sec_filings
  - query_macro 一次（全局共享）

回放时对参数做宽松匹配（见 _replay_*）：新闻按 max_items 截断、
申报按 form_types 过滤、宏观按 indicators 过滤；行情固定使用录制时的 period。
快照中没有的公司返回 error，不会穿透到真实接口。

**可见信息对齐**：非目标公司的新闻一律截断到 neighbor_max_items
（默认 config.KG_NEWS_PER_COMPANY = 5）。否则 B 组可以自行索要 8 条，
而 C 组的图谱工具固定取 5 条，两组看到的数据就不一样了，B↔C 的比较会失真。
人工标注的候选池也按同一口径生成（见 eval/gold.py）。

命令行：
  python -m eval.snapshot --out eval/snapshots/2026-09-20
  python -m eval.snapshot --inspect eval/snapshots/2026-09-20/snapshot.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import config
from live_kg.graph import CompanyGraph
from live_kg.queries import find_neighbors
from live_kg.seed import load_seed_relations
from tools import macro_tools, market_tools, news_tools, sec_tools
from tools.ticker_map import resolve_ticker

MACRO_KEY = "query_macro"


def entry_key(tool: str, ticker: Optional[str] = None) -> str:
    return tool if ticker is None else f"{tool}:{ticker.upper()}"


def seed_graph() -> CompanyGraph:
    graph = CompanyGraph()
    rows, errors = load_seed_relations()
    if errors:
        raise ValueError(f"种子关系文件有问题：{errors}")
    graph.load_seed(rows)
    return graph


def universe_for(targets: list[str], graph: Optional[CompanyGraph] = None) -> tuple[list[str], dict]:
    """目标公司 + 其 1~2 跳邻居，以及每家目标的邻居明细"""
    graph = graph or seed_graph()
    # path 里的三元组统一转成 list，保证 JSON 存取前后结构一致
    neighbors = {
        t: [{**n, "path": [list(edge) for edge in n["path"]]} for n in find_neighbors(graph, t)]
        for t in targets
    }
    universe = set(targets)
    for nbs in neighbors.values():
        universe.update(n["ticker"] for n in nbs)
    return sorted(universe), neighbors


class SnapshotStore:
    """
    快照的读写与回放。

    >>> store = SnapshotStore.load("eval/snapshots/2026-09-20/snapshot.json")
    >>> impls = store.tool_impls(target="AAPL")        # 注入 OrchestratorLoop
    >>> store.articles("TSM")                          # 生成标注候选池用
    """

    def __init__(self, data: dict):
        self.data = data

    # ── 读写 ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotStore":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    @property
    def targets(self) -> list[str]:
        return self.data.get("targets", [])

    @property
    def universe(self) -> list[str]:
        return self.data.get("universe", [])

    def neighbors(self, target: str) -> list[dict]:
        return self.data.get("neighbors", {}).get(target.upper(), [])

    def raw(self, tool: str, ticker: Optional[str] = None) -> Optional[dict]:
        entry = self.data.get("entries", {}).get(entry_key(tool, ticker))
        return entry.get("result") if entry else None

    def articles(self, ticker: str) -> list[dict]:
        result = self.raw("query_news", ticker) or {}
        return result.get("articles", []) if "error" not in result else []

    def stats(self) -> dict:
        entries = self.data.get("entries", {})
        failed = [k for k, e in entries.items() if "error" in (e.get("result") or {})]
        return {
            "recorded_at": self.data.get("recorded_at"),
            "targets": len(self.targets),
            "universe": len(self.universe),
            "entries": len(entries),
            "failed_entries": failed,
            "articles_total": sum(len(self.articles(t)) for t in self.universe),
        }

    # ── 录制 ────────────────────────────────────────────────────

    @classmethod
    def record(
        cls,
        targets: list[str],
        market_fn: Callable = market_tools.query_market_data,
        news_fn: Callable = news_tools.query_news,
        sec_fn: Callable = sec_tools.query_sec_filings,
        macro_fn: Callable = macro_tools.query_macro,
        fred_api_key: Optional[str] = None,
        on_progress: Callable[[str], None] = print,
    ) -> "SnapshotStore":
        targets = [t.upper() for t in targets]
        universe, neighbors = universe_for(targets)
        entries: dict[str, dict] = {}

        def _put(tool: str, ticker: Optional[str], params: dict, result: dict):
            entries[entry_key(tool, ticker)] = {
                "tool": tool, "ticker": ticker, "params": params, "result": result,
            }
            flag = "❌" if "error" in result else "✅"
            on_progress(f"  {flag} {entry_key(tool, ticker)}")

        on_progress(f"[snapshot] universe {len(universe)} 家：{', '.join(universe)}")
        for ticker in universe:
            period = config.MARKET_DEFAULT_PERIOD
            _put("query_market_data", ticker, {"period": period},
                 market_fn(ticker, period=period))
            max_items = config.NEWS_MAX_ITEMS
            _put("query_news", ticker, {"max_items": max_items},
                 news_fn(ticker, max_items=max_items))

        for ticker in targets:
            _put("query_sec_filings", ticker, {}, sec_fn(ticker))

        _put(MACRO_KEY, None, {}, macro_fn(config.get_fred_api_key(fred_api_key)))

        return cls({
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "targets": targets,
            "universe": universe,
            "neighbors": neighbors,
            "entries": entries,
        })

    # ── 回放 ────────────────────────────────────────────────────

    def _resolve(self, tool: str, entity: str, restrict_to: Optional[str]) -> tuple[Optional[str], Optional[dict]]:
        resolved = resolve_ticker(entity or "")
        if "error" in resolved:
            return None, resolved
        ticker = resolved["ticker"]
        if restrict_to and ticker != restrict_to.upper():
            return None, {"error": f"本组只分析目标公司 {restrict_to.upper()}，不支持查询 {ticker}"}
        if self.raw(tool, ticker) is None:
            return None, {"error": f"快照中没有 {ticker} 的 {tool} 数据（回放模式，不访问外部接口）"}
        return ticker, None

    def replay_market(self, entity: str, period: Optional[str] = None,
                      restrict_to: Optional[str] = None) -> dict:
        ticker, err = self._resolve("query_market_data", entity, restrict_to)
        if err:
            return err
        return json.loads(json.dumps(self.raw("query_market_data", ticker)))   # 深拷贝，防止被调用方改动

    def replay_news(self, entity: str, max_items: Optional[int] = None,
                    restrict_to: Optional[str] = None) -> dict:
        ticker, err = self._resolve("query_news", entity, restrict_to)
        if err:
            return err
        result = json.loads(json.dumps(self.raw("query_news", ticker)))
        if "error" in result:
            return result
        if max_items:
            result["articles"] = result.get("articles", [])[:int(max_items)]
            result["article_count"] = len(result["articles"])
        return result

    def replay_sec(self, entity: str, form_types: Optional[list[str]] = None,
                   restrict_to: Optional[str] = None) -> dict:
        ticker, err = self._resolve("query_sec_filings", entity, restrict_to)
        if err:
            return err
        result = json.loads(json.dumps(self.raw("query_sec_filings", ticker)))
        if "error" in result or not form_types:
            return result
        wanted = {f.upper() for f in form_types}
        result["filings"] = [f for f in result.get("filings", []) if str(f.get("form", "")).upper() in wanted]
        return result

    def replay_macro(self, indicators: Optional[list[str]] = None) -> dict:
        result = json.loads(json.dumps(self.raw(MACRO_KEY) or
                                       {"error": "快照中没有宏观数据（回放模式）"}))
        if "error" in result or not indicators:
            return result
        wanted = set(indicators)
        result["indicators"] = {k: v for k, v in result.get("indicators", {}).items() if k in wanted}
        result["indicator_docs"] = {k: v for k, v in result.get("indicator_docs", {}).items() if k in wanted}
        result["summary_text"] = macro_tools._format_indicators(result["indicators"])
        return result

    def news_fetcher(self) -> Callable:
        """供 tools.kg_live_tools.query_company_graph 使用的取数函数"""
        return lambda ticker, max_items: self.replay_news(ticker, max_items=max_items)

    def news_budget(self, entity: str, requested, target: Optional[str],
                    neighbor_max_items: int) -> Optional[int]:
        """非目标公司的新闻统一截断到 neighbor_max_items，保证各变体可见信息一致"""
        resolved = resolve_ticker(entity or "")
        ticker = resolved.get("ticker")
        if target and ticker and ticker != target.upper():
            return min(int(requested), neighbor_max_items) if requested else neighbor_max_items
        return requested

    def tool_impls(self, target: Optional[str] = None, restrict_to: Optional[str] = None,
                   extractor=None, neighbor_max_items: Optional[int] = None) -> dict:
        """
        生成注入 OrchestratorLoop 的工具实现（签名统一为 (tool_input, tool_log)）。

        restrict_to        : 只允许查询该 ticker（A 组用，防止基线组"偷看"邻居）
        extractor          : 知识图谱事件抽取器；为 None 时不提供 query_company_graph
        neighbor_max_items : 非目标公司最多返回几条新闻，默认 config.KG_NEWS_PER_COMPANY
        """
        neighbor_max_items = neighbor_max_items or config.KG_NEWS_PER_COMPANY
        impls = {
            "query_market_data": lambda ti, tl: self.replay_market(
                ti.get("entity", ""), ti.get("period"), restrict_to),
            "query_news": lambda ti, tl: self.replay_news(
                ti.get("entity", ""),
                self.news_budget(ti.get("entity", ""), ti.get("max_items"), target, neighbor_max_items),
                restrict_to),
            "query_sec_filings": lambda ti, tl: self.replay_sec(
                ti.get("entity", ""), ti.get("form_types"), restrict_to),
            "query_macro": lambda ti, tl: self.replay_macro(ti.get("indicators")),
        }
        if extractor is not None:
            from tools import kg_live_tools

            def _graph(ti, tl):
                prior = {}
                for entry in tl:
                    result = entry.get("result") or {}
                    if entry.get("tool") in ("query_market_data", "query_news", "query_sec_filings") \
                            and "error" not in result:
                        prior[entry["tool"]] = result
                return kg_live_tools.query_company_graph(
                    ti.get("entity", ""), extractor=extractor, prior_results=prior,
                    news_fetcher=lambda t, n: self.replay_news(
                        t, self.news_budget(t, n, target, neighbor_max_items)),
                )

            impls["query_company_graph"] = _graph
        return impls


# ── 命令行 ───────────────────────────────────────────────────────

def load_companies(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip().upper() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def main(argv=None):
    parser = argparse.ArgumentParser(description="录制 / 查看实验用的数据快照")
    parser.add_argument("--companies", default="eval/companies.txt")
    parser.add_argument("--out", help="输出目录，默认 eval/snapshots/<今天>")
    parser.add_argument("--inspect", help="查看已有快照的统计信息")
    args = parser.parse_args(argv)

    if args.inspect:
        print(json.dumps(SnapshotStore.load(args.inspect).stats(), ensure_ascii=False, indent=2))
        return

    targets = load_companies(args.companies)
    out_dir = Path(args.out or f"eval/snapshots/{datetime.now().strftime('%Y-%m-%d')}")
    store = SnapshotStore.record(targets)
    path = store.save(out_dir / "snapshot.json")

    stats = store.stats()
    print(f"\n[snapshot] 已保存：{path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["failed_entries"]:
        print("\n⚠️  以下调用失败（回放时这些工具会返回同样的 error，属于正常记录）：")
        for k in stats["failed_entries"]:
            print(f"   - {k}")


if __name__ == "__main__":
    main()
