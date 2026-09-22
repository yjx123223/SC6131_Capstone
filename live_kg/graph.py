"""
live_kg/graph.py
-----------------
CompanyGraph：带本体校验的公司知识图谱构建器（底层是 networkx.MultiDiGraph）。

职责：
  - 按本体创建节点（同一 ID 重复添加时合并属性，类型冲突则拒绝）
  - 写边前调用 ontology.validate_edge 校验两端类型，不合法的边被拒绝并记录
  - 事件去重：同一类型、同一批受影响公司、同一天、摘要前缀相同 → 合并为一个事件，
    证据（REPORTED_BY）累加，置信度取较大值
  - 导出：to_dict()（JSON 存档）、to_mermaid()（报告中的关系图）

约定：MultiDiGraph 的边 key 就是关系名，所以同一对节点之间同一种关系只有一条边；
对称关系（COMPETES_WITH / PARTNERS_WITH）只存一条，查询时两个方向都看。
"""

import hashlib
import re
from typing import Iterable, Optional

import networkx as nx

from . import ontology as ont


class CompanyGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.rejected: list[str] = []      # 被本体校验拒绝的写入
        self._event_seq = 0
        self._event_keys: dict[tuple, str] = {}

    # ── 节点 ────────────────────────────────────────────────────

    def _add_node(self, node_type: str, key: str, **attrs) -> str:
        nid = ont.node_id(node_type, key)
        clean = {k: v for k, v in attrs.items() if v is not None}
        if nid in self.g:
            existing = self.g.nodes[nid].get("type")
            if existing != node_type:
                raise ValueError(f"节点 {nid} 类型冲突：{existing} vs {node_type}")
            self.g.nodes[nid].update(clean)
        else:
            self.g.add_node(nid, type=node_type, **clean)
        return nid

    def add_company(self, ticker: str, name: Optional[str] = None, is_target: bool = False) -> str:
        ticker = ticker.upper()
        nid = self._add_node(ont.COMPANY, ticker, ticker=ticker, name=name)
        if is_target:
            self.g.nodes[nid]["is_target"] = True
        self.g.nodes[nid].setdefault("is_target", False)
        return nid

    def add_industry(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"
        return self._add_node(ont.INDUSTRY, slug, name=name)

    def add_filing(self, form: str, filing_date: str, url: str = "", accession: str = "") -> str:
        key = accession or hashlib.sha1(f"{form}|{filing_date}|{url}".encode()).hexdigest()[:12]
        return self._add_node(ont.FILING, key, form=form, filing_date=filing_date, url=url)

    def add_news(self, title: str, publisher: str = "", published_at: str = "", url: str = "") -> str:
        basis = url or title.strip().lower()
        key = hashlib.sha1(basis.encode()).hexdigest()[:12]
        return self._add_node(
            ont.NEWS, key, title=title, publisher=publisher, published_at=published_at, url=url
        )

    def add_event(
        self,
        event_type: str,
        polarity: str,
        date: str,
        summary: str,
        affected: Iterable[str],
        evidence: Iterable[str],
        confidence: float = 0.5,
    ) -> Optional[str]:
        """
        添加事件节点及其 AFFECTS / REPORTED_BY 边。

        affected : 受影响公司的 ticker（必须已经是图中的 Company 节点）
        evidence : NewsArticle 节点 ID
        返回事件节点 ID；事件类型/正负面不合法或没有任何受影响公司时返回 None。
        """
        if event_type not in ont.EVENT_TYPES:
            self.rejected.append(f"事件类型不合法：{event_type}")
            return None
        if polarity not in ont.POLARITIES:
            self.rejected.append(f"事件正负面不合法：{polarity}")
            return None

        company_ids = []
        for t in affected:
            cid = ont.node_id(ont.COMPANY, t.upper())
            if cid in self.g:
                company_ids.append(cid)
            else:
                self.rejected.append(f"事件关联的公司不在图中：{t}")
        if not company_ids:
            return None

        confidence = max(0.0, min(1.0, float(confidence)))
        norm_summary = re.sub(r"\s+", "", summary.lower())[:30]
        dedupe_key = (event_type, frozenset(company_ids), date, norm_summary)

        if dedupe_key in self._event_keys:
            eid = self._event_keys[dedupe_key]
            node = self.g.nodes[eid]
            node["confidence"] = max(node["confidence"], confidence)
        else:
            self._event_seq += 1
            label = f"E{self._event_seq}"
            eid = self._add_node(
                ont.EVENT, label, label=label, event_type=event_type, polarity=polarity,
                date=date, summary=summary, confidence=confidence,
            )
            self._event_keys[dedupe_key] = eid
            for cid in company_ids:
                self.add_relation("AFFECTS", eid, cid, polarity=polarity)

        for nid in evidence:
            self.add_relation("REPORTED_BY", eid, nid)
        return eid

    # ── 边 ──────────────────────────────────────────────────────

    def add_relation(self, relation: str, src: str, dst: str, **attrs) -> bool:
        if src not in self.g or dst not in self.g:
            self.rejected.append(f"{relation}：节点不存在（{src} → {dst}）")
            return False
        err = ont.validate_edge(relation, self.g.nodes[src]["type"], self.g.nodes[dst]["type"])
        if err:
            self.rejected.append(err)
            return False
        if ont.is_symmetric(relation) and self.g.has_edge(dst, src, key=relation):
            src, dst = dst, src          # 对称关系只保留一条边
        clean = {k: v for k, v in attrs.items() if v is not None}
        if self.g.has_edge(src, dst, key=relation):
            self.g.edges[src, dst, relation].update(clean)
        else:
            self.g.add_edge(src, dst, key=relation, relation=relation, **clean)
        return True

    def add_company_relation(self, subject: str, relation: str, obj: str, **attrs) -> bool:
        """两端用 ticker 表示的公司间关系（种子数据用）"""
        s = self.add_company(subject)
        o = self.add_company(obj)
        return self.add_relation(relation, s, o, **attrs)

    def load_seed(self, rows: Iterable[dict]) -> int:
        n = 0
        for r in rows:
            if self.add_company_relation(r["subject"], r["relation"], r["object"],
                                         note=r.get("note", ""), source="seed"):
                n += 1
        return n

    # ── 查询辅助 ────────────────────────────────────────────────

    def has_company(self, ticker: str) -> bool:
        return ont.node_id(ont.COMPANY, ticker.upper()) in self.g

    def company_edges(self, ticker: str) -> list[tuple[str, str, str]]:
        """与某公司相连的公司间关系：[(subject_ticker, relation, object_ticker)]"""
        cid = ont.node_id(ont.COMPANY, ticker.upper())
        if cid not in self.g:
            return []
        out = []
        for u, v, rel in list(self.g.out_edges(cid, keys=True)) + list(self.g.in_edges(cid, keys=True)):
            if self.g.nodes[u]["type"] == ont.COMPANY and self.g.nodes[v]["type"] == ont.COMPANY:
                out.append((self.g.nodes[u]["ticker"], rel, self.g.nodes[v]["ticker"]))
        return out

    def events_for(self, ticker: str) -> list[dict]:
        """影响某公司的所有事件（附证据新闻），按日期倒序"""
        cid = ont.node_id(ont.COMPANY, ticker.upper())
        if cid not in self.g:
            return []
        events = []
        for eid, _, rel in self.g.in_edges(cid, keys=True):
            if rel != "AFFECTS":
                continue
            node = self.g.nodes[eid]
            sources = [
                {k: self.g.nodes[n].get(k, "") for k in ("title", "publisher", "published_at", "url")}
                for _, n, r in self.g.out_edges(eid, keys=True) if r == "REPORTED_BY"
            ]
            events.append({
                "id": node["label"],
                "event_type": node["event_type"],
                "polarity": node["polarity"],
                "date": node["date"],
                "summary": node["summary"],
                "confidence": node["confidence"],
                "sources": sources,
            })
        events.sort(key=lambda e: (e["date"], e["id"]), reverse=True)
        return events

    def event_ids(self) -> set[str]:
        return {d["label"] for _, d in self.g.nodes(data=True) if d["type"] == ont.EVENT}

    def prune_companies(self, keep: Iterable[str]):
        """删除不在 keep 里的公司节点，以及因此变成孤立的事件/新闻/行业节点"""
        keep_ids = {ont.node_id(ont.COMPANY, t.upper()) for t in keep}
        drop = [n for n, d in self.g.nodes(data=True) if d["type"] == ont.COMPANY and n not in keep_ids]
        self.g.remove_nodes_from(drop)
        # 不再影响任何公司的事件
        dead_events = [
            n for n, d in self.g.nodes(data=True)
            if d["type"] == ont.EVENT
            and not any(r == "AFFECTS" for _, _, r in self.g.out_edges(n, keys=True))
        ]
        self.g.remove_nodes_from(dead_events)
        for key in [k for k, v in self._event_keys.items() if v in dead_events]:
            del self._event_keys[key]
        # 因此变成孤立的新闻 / 行业 / 申报节点
        orphans = [n for n in self.g.nodes if self.g.degree(n) == 0 and self.g.nodes[n]["type"] != ont.COMPANY]
        self.g.remove_nodes_from(orphans)

    def stats(self) -> dict:
        nodes: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            nodes[d["type"]] = nodes.get(d["type"], 0) + 1
        edges: dict[str, int] = {}
        for _, _, rel in self.g.edges(keys=True):
            edges[rel] = edges.get(rel, 0) + 1
        return {
            "node_count": self.g.number_of_nodes(),
            "edge_count": self.g.number_of_edges(),
            "nodes_by_type": nodes,
            "edges_by_relation": edges,
            "rejected_count": len(self.rejected),
        }

    # ── 导出 ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n, **d} for n, d in self.g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)],
            "stats": self.stats(),
        }

    def to_mermaid(self, max_events_per_company: int = 3, highlight_events: Iterable[str] = ()) -> str:
        """
        生成 Mermaid 关系图：公司、行业、事件三类节点（新闻/申报不画，避免太乱）。
        每家公司最多画 max_events_per_company 个事件，highlight_events 中的事件优先。
        """
        highlight = set(highlight_events)
        ids: dict[str, str] = {}

        def mid(n):
            if n not in ids:
                ids[n] = f"n{len(ids)}"
            return ids[n]

        def esc(text):
            return str(text).replace('"', "'").replace("\n", " ")

        lines = ["graph LR"]
        for n, d in self.g.nodes(data=True):
            if d["type"] == ont.COMPANY:
                name = d.get("name") or d["ticker"]
                label = d["ticker"] if name == d["ticker"] else f"{name} ({d['ticker']})"
                cls = ":::target" if d.get("is_target") else ":::company"
                lines.append(f'  {mid(n)}["{esc(label)}"]{cls}')
            elif d["type"] == ont.INDUSTRY:
                lines.append(f'  {mid(n)}[/"{esc(d["name"])}"/]:::industry')

        drawn_events = set()
        for n, d in self.g.nodes(data=True):
            if d["type"] != ont.COMPANY:
                continue
            evs = [e for e, _, r in self.g.in_edges(n, keys=True) if r == "AFFECTS"]
            evs.sort(key=lambda e: (self.g.nodes[e]["label"] not in highlight,
                                    -self.g.nodes[e]["confidence"]))
            drawn_events.update(evs[:max_events_per_company])

        for e in sorted(drawn_events, key=lambda x: int(self.g.nodes[x]["label"][1:])):
            d = self.g.nodes[e]
            arrow = {"positive": "▲", "negative": "▼"}.get(d["polarity"], "●")
            text = f"{d['label']} {ont.EVENT_TYPES.get(d['event_type'], d['event_type'])} {arrow}"
            lines.append(f'  {mid(e)}(["{esc(text)}"]):::{d["polarity"]}')

        for u, v, rel in self.g.edges(keys=True):
            tu, tv = self.g.nodes[u]["type"], self.g.nodes[v]["type"]
            if rel == "AFFECTS" and u in drawn_events:
                lines.append(f"  {mid(u)} -.-> {mid(v)}")
            elif rel in ("SUPPLIES_TO", "IN_INDUSTRY"):
                lines.append(f"  {mid(u)} -->|{rel}| {mid(v)}")
            elif rel in ("COMPETES_WITH", "PARTNERS_WITH") and tu == tv == ont.COMPANY:
                lines.append(f"  {mid(u)} <-->|{rel}| {mid(v)}")

        lines += [
            "  classDef target fill:#1f6feb,color:#fff,stroke:#1f6feb",
            "  classDef company fill:#eef2f7,stroke:#8b98a9",
            "  classDef industry fill:#f5f0e6,stroke:#b59f73",
            "  classDef positive fill:#e3f5e8,stroke:#2e8b57",
            "  classDef negative fill:#fde8e8,stroke:#c0392b",
            "  classDef neutral fill:#f0f0f0,stroke:#999",
        ]
        return "\n".join(lines)
