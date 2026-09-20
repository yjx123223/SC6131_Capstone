"""
eval/metrics.py
----------------
把 eval/runs（报告）+ eval/gold（标注）+ eval/results/judgements（判定）
汇总成指标表。口径见 docs/eval-design.md 5.2。

| 指标 | 含义 | 来源 |
|------|------|------|
| **M1** | 跨公司风险召回率（**主指标**）：material 且 hop≥1 的事件被提及的比例 | 判定器 |
| M2  | 2 跳召回率（hop=2 子集） | 判定器 |
| M2b | 目标公司事件召回率（hop=0，对照用：三组数据相同，不该有大差异） | 判定器 |
| M3  | 引用可核验率：可核对的论断中判为 supported 的比例（judgment 类不计入分母） | 判定器 |
| M4  | 无依据率：同一分母下判为 unsupported 的比例；另加确定性检查（编造的事件编号） | 判定器 + 代码 |
| M5  | 成本：调用次数、token、耗时 | 结果文件里的 usage |
| M6  | 稳定性：3 次重复之间，配置建议是否一致 + 提及事件集合的平均 Jaccard | 纯代码 |

统计口径：
  - 以**公司**为配对单位（n=15），3 次重复先取中位数，再做变体间比较。
  - 某公司若没有 material 且 hop≥1 的事件，M1 分母为 0，该公司**整体排除**出 M1
    （三个变体一起排除，保持配对）。排除了几家会在输出里写明。
  - 显著性检验放在 eval/report.py（E5），这里只出数。

命令行：
  python -m eval.metrics                      # 打印汇总 + 写 eval/results/*.csv
  python -m eval.metrics --by-company         # 打印每家公司的 M1 明细
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from eval import gold as gold_mod
from eval.judge import DEFAULT_JUDGEMENTS, load_judgements
from eval.runner import DEFAULT_OUT, load_runs

DEFAULT_RESULTS = "eval/results"
_EVENT_ID_PREFIX = "E"


# ── gold 的切片 ──────────────────────────────────────────────────

def gold_slices(gold_rows: list[dict]) -> dict:
    """company → {"cross": {id...}, "hop2": {...}, "hop0": {...}}（都限定 material=1）"""
    slices: dict[str, dict[str, set]] = defaultdict(lambda: {"cross": set(), "hop2": set(), "hop0": set()})
    for r in gold_rows:
        if str(r.get("material", "")).strip() != "1":
            continue
        company, hop, eid = r["company"], int(r.get("hop") or 0), int(r["id"])
        if hop >= 1:
            slices[company]["cross"].add(eid)
        if hop == 2:
            slices[company]["hop2"].add(eid)
        if hop == 0:
            slices[company]["hop0"].add(eid)
    return dict(slices)


def recall(mentioned: set[int], target: set[int]) -> Optional[float]:
    """分母为 0 时返回 None——不是 0 分，是"这家公司没有这类事件可召回" """
    if not target:
        return None
    return len(mentioned & target) / len(target)


# ── 单份报告的指标 ───────────────────────────────────────────────

def cited_event_ids(draft: dict) -> set[str]:
    return {str(x).strip().upper() for x in (draft.get("cited_event_ids") or []) if str(x).strip()}


def graph_event_ids(tool_log: list) -> set[str]:
    """这份报告当时真正拿到的图谱事件编号（用于查编造的引用）"""
    ids = set()
    for entry in tool_log or []:
        if entry.get("tool") != "query_company_graph":
            continue
        result = entry.get("result") or {}
        for key in ("target_events", "propagated_risks", "opportunities"):
            for item in result.get(key) or []:
                eid = item.get("id") or item.get("event_id")
                if eid:
                    ids.add(str(eid).strip().upper())
    return ids


def run_metrics(run: dict, judgement: dict, slices: dict, count_unverified: bool = False) -> dict:
    """count_unverified=True 时把证据未核实的提及也算进召回（用于看这层过滤的影响）"""
    company = run["company"]
    sl = slices.get(company, {"cross": set(), "hop2": set(), "hop0": set()})
    raw = judgement.get("mentions", {}).get("mentioned", [])
    # 证据不在报告里的判定是假阳性（判定器摘了候选事件的原文），默认不计入
    unverified = [m for m in raw if not m.get("verified", True)]
    kept = raw if count_unverified else [m for m in raw if m.get("verified", True)]
    mentioned = {m["id"] for m in kept}

    verdicts = judgement.get("citations", {}).get("verdicts", [])
    checkable = [v for v in verdicts if v["verdict"] in ("supported", "unsupported")]
    supported = sum(1 for v in checkable if v["verdict"] == "supported")

    usage = run.get("usage") or {}
    cited, available = cited_event_ids(run.get("draft") or {}), graph_event_ids(run.get("tool_log"))

    return {
        "variant": run["variant"], "company": company, "repeat": run["repeat"],
        "m1_cross_recall": recall(mentioned, sl["cross"]),
        "m2_hop2_recall": recall(mentioned, sl["hop2"]),
        "m2b_hop0_recall": recall(mentioned, sl["hop0"]),
        "cross_hit": len(mentioned & sl["cross"]), "cross_total": len(sl["cross"]),
        "hop2_hit": len(mentioned & sl["hop2"]), "hop2_total": len(sl["hop2"]),
        "hop0_hit": len(mentioned & sl["hop0"]), "hop0_total": len(sl["hop0"]),
        "mentioned_ids": sorted(mentioned),
        "mentions_unverified": len(unverified),
        "m3_citation_supported": supported / len(checkable) if checkable else None,
        "m4_unsupported": (len(checkable) - supported) / len(checkable) if checkable else None,
        "claims_checkable": len(checkable), "claims_judgment": len(verdicts) - len(checkable),
        # 确定性检查：报告引用了图谱里不存在的事件编号
        "fabricated_event_ids": sorted(cited - available) if available or cited else [],
        "m5_calls": usage.get("calls"), "m5_input_tokens": usage.get("input_tokens"),
        "m5_output_tokens": usage.get("output_tokens"), "m5_seconds": run.get("wall_seconds"),
        "recommendation": (run.get("draft") or {}).get("recommendation"),
        "confidence": (run.get("draft") or {}).get("confidence"),
    }


def build_rows(runs: list[dict], judgements: list[dict], gold_rows: list[dict],
               count_unverified: bool = False) -> list[dict]:
    slices = gold_slices(gold_rows)
    index = {(j["variant"], j["company"], j["repeat"]): j for j in judgements}
    rows, missing = [], []
    for run in runs:
        key = (run["variant"], run["company"], run["repeat"])
        if key not in index:
            missing.append(key)
            continue
        rows.append(run_metrics(run, index[key], slices, count_unverified))
    if missing:
        raise SystemExit(f"有 {len(missing)} 份结果还没判定（如 {missing[:3]}），先跑 python -m eval.judge")
    return rows


# ── M6 稳定性 ────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stability(rows: list[dict]) -> dict:
    """(variant, company) → {建议一致, 提及集合平均 Jaccard}"""
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["company"])].append(r)
    out = {}
    for key, group in by.items():
        group = sorted(group, key=lambda r: r["repeat"])
        recs = [r["recommendation"] for r in group]
        sets = [set(r["mentioned_ids"]) for r in group]
        pairs = [jaccard(sets[i], sets[j])
                 for i in range(len(sets)) for j in range(i + 1, len(sets))]
        out[key] = {
            "n": len(group),
            "rec_identical": len(set(recs)) == 1,
            "rec_majority": max(recs.count(x) for x in set(recs)) / len(recs) if recs else None,
            "mention_jaccard": st.mean(pairs) if pairs else None,
        }
    return out


# ── 汇总 ─────────────────────────────────────────────────────────

def _median(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return st.median(values) if values else None


def by_company(rows: list[dict]) -> dict:
    """(variant, company) → 3 次重复取中位数后的指标"""
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["company"])].append(r)
    keys = ["m1_cross_recall", "m2_hop2_recall", "m2b_hop0_recall", "m3_citation_supported",
            "m4_unsupported", "m5_calls", "m5_input_tokens", "m5_output_tokens", "m5_seconds"]
    return {k: {m: _median(r[m] for r in group) for m in keys} for k, group in by.items()}


def paired_companies(per_company: dict, metric: str, variants: list[str]) -> list[str]:
    """三个变体都有该指标值的公司——保持配对，缺一个就整家排除"""
    companies = sorted({c for (_, c) in per_company})
    return [c for c in companies
            if all(per_company.get((v, c), {}).get(metric) is not None for v in variants)]


def summarize(rows: list[dict], variants: list[str] = ("A", "B", "C")) -> dict:
    per_company = by_company(rows)
    stab = stability(rows)
    companies_all = sorted({c for (_, c) in per_company})
    paired = paired_companies(per_company, "m1_cross_recall", list(variants))

    summary = {"companies": companies_all, "m1_paired_companies": paired,
               "m1_excluded": [c for c in companies_all if c not in paired], "by_variant": {}}
    for v in variants:
        vals = lambda m, cs=None: [per_company[(v, c)][m] for c in (cs or companies_all)
                                   if per_company.get((v, c), {}).get(m) is not None]
        st_rows = [s for (vv, _), s in stab.items() if vv == v]
        m1 = vals("m1_cross_recall", paired)
        summary["by_variant"][v] = {
            "m1_mean": st.mean(m1) if m1 else None,
            "m1_median": st.median(m1) if m1 else None,
            "m1_sd": st.stdev(m1) if len(m1) > 1 else None,
            "m1_values": m1,
            "m2_mean": st.mean(vals("m2_hop2_recall")) if vals("m2_hop2_recall") else None,
            "m2b_mean": st.mean(vals("m2b_hop0_recall")) if vals("m2b_hop0_recall") else None,
            "m3_mean": st.mean(vals("m3_citation_supported")) if vals("m3_citation_supported") else None,
            "m4_mean": st.mean(vals("m4_unsupported")) if vals("m4_unsupported") else None,
            "m5_calls": st.mean(vals("m5_calls")) if vals("m5_calls") else None,
            "m5_tokens": (st.mean(vals("m5_input_tokens")) + st.mean(vals("m5_output_tokens")))
                         if vals("m5_input_tokens") else None,
            "m5_seconds": st.mean(vals("m5_seconds")) if vals("m5_seconds") else None,
            "m6_rec_identical": st.mean([1.0 if s["rec_identical"] else 0.0 for s in st_rows]) if st_rows else None,
            "m6_jaccard": st.mean([s["mention_jaccard"] for s in st_rows
                                   if s["mention_jaccard"] is not None]) if st_rows else None,
            "fabricated_runs": sum(1 for r in rows if r["variant"] == v and r["fabricated_event_ids"]),
            "mentions_unverified": sum(r["mentions_unverified"] for r in rows if r["variant"] == v),
        }
    return summary


def print_summary(summary: dict, on_progress: Callable[[str], None] = print):
    excluded = summary["m1_excluded"]
    on_progress(f"\n配对公司 {len(summary['m1_paired_companies'])} 家"
                + (f"（排除 {', '.join(excluded)}：没有 material 且 hop≥1 的事件）" if excluded else ""))
    head = (f"\n{'变体':<5}{'M1跨公司召回':>13}{'±sd':>8}{'M2两跳':>9}{'M2b本公司':>11}"
            f"{'M3可核验':>10}{'M4无依据':>10}{'M6建议一致':>11}{'M6Jaccard':>11}{'调用':>7}{'tokens':>9}")
    on_progress(head)
    for v, s in sorted(summary["by_variant"].items()):
        fmt = lambda x, pct=True: ("—" if x is None else (f"{x*100:.1f}%" if pct else f"{x:.2f}"))
        on_progress(f"{v:<5}{fmt(s['m1_mean']):>13}{fmt(s['m1_sd']):>8}{fmt(s['m2_mean']):>9}"
                    f"{fmt(s['m2b_mean']):>11}{fmt(s['m3_mean']):>10}{fmt(s['m4_mean']):>10}"
                    f"{fmt(s['m6_rec_identical']):>11}{fmt(s['m6_jaccard'], False):>11}"
                    f"{s['m5_calls']:>7.1f}{s['m5_tokens']:>9.0f}")
    fab = {v: s["fabricated_runs"] for v, s in summary["by_variant"].items() if s["fabricated_runs"]}
    on_progress(f"\n引用了图谱里不存在的事件编号：{fab or '无'}")
    unver = {v: s["mentions_unverified"] for v, s in summary["by_variant"].items() if s["mentions_unverified"]}
    on_progress(f"证据未能在报告里核实、已排除的提及：{unver or '无'}"
                "（--count-unverified 可看计入后的数字）")


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if k != "mentioned_ids"] + ["mentioned_ids"] if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["mentioned_ids"] = " ".join(str(x) for x in r["mentioned_ids"])
            out["fabricated_event_ids"] = " ".join(r["fabricated_event_ids"])
            writer.writerow(out)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="汇总消融实验指标")
    parser.add_argument("--runs", default=DEFAULT_OUT)
    parser.add_argument("--judgements", default=DEFAULT_JUDGEMENTS)
    parser.add_argument("--gold", default="eval/gold/gold_events.csv")
    parser.add_argument("--out", default=DEFAULT_RESULTS)
    parser.add_argument("--by-company", action="store_true", help="打印每家公司的 M1 明细")
    parser.add_argument("--count-unverified", action="store_true",
                        help="把证据未核实的提及也算进召回（默认排除）")
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    judgements = load_judgements(args.judgements)
    gold_rows = gold_mod.read_csv(args.gold)
    if not runs or not judgements:
        raise SystemExit("缺少结果或判定：先跑 eval.runner，再跑 eval.judge")

    rows = build_rows(runs, judgements, gold_rows, args.count_unverified)
    summary = summarize(rows)
    print_summary(summary)

    if args.by_company:
        per = by_company(rows)
        print(f"\n{'公司':<7}" + "".join(f"{v:>10}" for v in ("A", "B", "C")) + "   （M1 中位数）")
        for c in summary["companies"]:
            cells = []
            for v in ("A", "B", "C"):
                x = per.get((v, c), {}).get("m1_cross_recall")
                cells.append("—" if x is None else f"{x*100:.0f}%")
            mark = "" if c in summary["m1_paired_companies"] else "  (未计入)"
            print(f"{c:<7}" + "".join(f"{x:>10}" for x in cells) + mark)

    out_dir = Path(args.out)
    write_csv(rows, out_dir / "metrics.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out_dir / 'metrics.csv'} 与 {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
