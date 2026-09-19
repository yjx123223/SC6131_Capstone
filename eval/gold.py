"""
eval/gold.py
-------------
从快照生成人工标注用的候选事件池，并对标注结果做校验与统计。

候选池口径（与各变体可见的数据一致）：
  目标公司自己的新闻      最多 config.NEWS_MAX_ITEMS 条（hop=0）
  每个 1~2 跳邻居的新闻   最多 config.KG_NEWS_PER_COMPANY 条（hop=1/2）

标注规范见 docs/annotation-guide.md。标注者只填 material / impact_on_target / note 三列。

命令行：
  python -m eval.gold --snapshot eval/snapshots/2026-09-19/snapshot.json    # 生成待标注 CSV
  python -m eval.gold --check eval/gold/gold_events.csv                     # 校验 + 统计
  python -m eval.gold --consistency 第一次.csv 第二次.csv                    # 自一致率
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import config
from eval.snapshot import SnapshotStore
from live_kg.queries import ROLE_ZH

COLUMNS = ["id", "company", "source_company", "hop", "relation",
           "published_at", "publisher", "title", "summary", "url",
           "material", "impact_on_target", "note"]

FILLED_BY_ANNOTATOR = ["material", "impact_on_target", "note"]
VALID_IMPACT = {"positive", "negative", "neutral", ""}


def build_rows(store: SnapshotStore, target_max_items: Optional[int] = None,
               neighbor_max_items: Optional[int] = None) -> list[dict]:
    """生成候选池：按 company → hop → source_company → 时间倒序排列"""
    target_max_items = target_max_items or config.NEWS_MAX_ITEMS
    neighbor_max_items = neighbor_max_items or config.KG_NEWS_PER_COMPANY

    rows: list[dict] = []
    for target in store.targets:
        sources = [(target, 0, "目标公司", target_max_items)]
        for n in store.neighbors(target):
            label = "、".join(ROLE_ZH[r] for r in n["roles"])
            if n["hop"] == 2 and n.get("via"):
                label += f"（经由 {n['via']}）"
            sources.append((n["ticker"], n["hop"], label, neighbor_max_items))

        for source_company, hop, relation, limit in sources:
            articles = sorted(store.articles(source_company),
                              key=lambda a: a.get("published_at", ""), reverse=True)[:limit]
            for a in articles:
                rows.append({
                    "id": 0,
                    "company": target,
                    "source_company": source_company,
                    "hop": hop,
                    "relation": relation,
                    "published_at": str(a.get("published_at", ""))[:10],
                    "publisher": a.get("publisher", ""),
                    "title": a.get("title", ""),
                    "summary": (a.get("summary", "") or "")[:200],
                    "url": a.get("url", ""),
                    "material": "",
                    "impact_on_target": "",
                    "note": "",
                })
    for i, row in enumerate(rows, 1):
        row["id"] = i
    return rows


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig：Excel 打开不乱码
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check(rows: list[dict]) -> dict:
    """
    校验标注结果并统计。

    Returns {"errors": [...], "warnings": [...], "summary": {...}, "by_company": {...}}
    """
    errors, warnings = [], []
    labelled = 0
    by_company: dict[str, dict] = {}

    for row in rows:
        rid = row.get("id", "?")
        material = (row.get("material") or "").strip()
        impact = (row.get("impact_on_target") or "").strip().lower()
        company = row.get("company", "?")
        stats = by_company.setdefault(company, {"total": 0, "labelled": 0, "material": 0,
                                                "hop0": 0, "hop1": 0, "hop2": 0})
        stats["total"] += 1

        if material == "":
            continue
        if material not in ("0", "1"):
            errors.append(f"#{rid} material 只能是 0 或 1，当前是 {material!r}")
            continue

        labelled += 1
        stats["labelled"] += 1
        if impact not in VALID_IMPACT:
            errors.append(f"#{rid} impact_on_target 只能是 positive/negative/neutral，当前是 {impact!r}")
        if material == "1":
            stats["material"] += 1
            stats[f"hop{row.get('hop', 0)}"] = stats.get(f"hop{row.get('hop', 0)}", 0) + 1
            if impact == "":
                errors.append(f"#{rid} material=1 必须填 impact_on_target")
        elif impact:
            warnings.append(f"#{rid} material=0 却填了 impact_on_target，将被忽略")

    for company, s in by_company.items():
        if s["labelled"] == s["total"] and s["material"] == 0:
            warnings.append(f"{company} 没有任何 material=1 的事件，该公司会被排除在召回率之外")
        if s["labelled"] and s["labelled"] < s["total"]:
            warnings.append(f"{company} 只标了 {s['labelled']}/{s['total']} 条")

    total_material = sum(s["material"] for s in by_company.values())
    return {
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "rows": len(rows),
            "labelled": labelled,
            "material": total_material,
            "material_hop1_plus": sum(s.get("hop1", 0) + s.get("hop2", 0) for s in by_company.values()),
            "companies": len(by_company),
        },
        "by_company": by_company,
    }


def consistency(first: list[dict], second: list[dict]) -> dict:
    """两次标注的自一致率（只比较两边都标了的 id）"""
    a = {r["id"]: (r.get("material", "").strip(), (r.get("impact_on_target") or "").strip().lower())
         for r in first if (r.get("material") or "").strip()}
    b = {r["id"]: (r.get("material", "").strip(), (r.get("impact_on_target") or "").strip().lower())
         for r in second if (r.get("material") or "").strip()}
    common = sorted(set(a) & set(b), key=lambda x: int(x) if str(x).isdigit() else 0)
    if not common:
        return {"compared": 0, "material_agreement": None, "full_agreement": None, "disagreements": []}

    material_same = [i for i in common if a[i][0] == b[i][0]]
    full_same = [i for i in common if a[i] == b[i]]
    return {
        "compared": len(common),
        "material_agreement": round(len(material_same) / len(common), 3),
        "full_agreement": round(len(full_same) / len(common), 3),
        "disagreements": [{"id": i, "first": a[i], "second": b[i]} for i in common if a[i] != b[i]],
    }


# ── 命令行 ───────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="生成 / 校验人工标注文件")
    parser.add_argument("--snapshot", help="快照路径，生成待标注 CSV")
    parser.add_argument("--out", default="eval/gold/gold_events.csv")
    parser.add_argument("--check", help="校验已标注的 CSV")
    parser.add_argument("--consistency", nargs=2, metavar=("FIRST", "SECOND"), help="计算两次标注的一致率")
    args = parser.parse_args(argv)

    if args.consistency:
        result = consistency(read_csv(args.consistency[0]), read_csv(args.consistency[1]))
        print(f"比较 {result['compared']} 条：material 一致率 {result['material_agreement']}，"
              f"含方向的完全一致率 {result['full_agreement']}")
        for d in result["disagreements"]:
            print(f"  #{d['id']}: {d['first']} → {d['second']}")
        return

    if args.check:
        result = check(read_csv(args.check))
        s = result["summary"]
        print(f"共 {s['rows']} 条，已标 {s['labelled']} 条；material=1 共 {s['material']} 条"
              f"（其中跨公司 {s['material_hop1_plus']} 条），覆盖 {s['companies']} 家公司\n")
        print(f"{'公司':<8}{'总数':>6}{'已标':>6}{'material':>10}{'hop0':>6}{'hop1':>6}{'hop2':>6}")
        for company, st in sorted(result["by_company"].items()):
            print(f"{company:<8}{st['total']:>6}{st['labelled']:>6}{st['material']:>10}"
                  f"{st.get('hop0', 0):>6}{st.get('hop1', 0):>6}{st.get('hop2', 0):>6}")
        for e in result["errors"]:
            print(f"❌ {e}")
        for w in result["warnings"]:
            print(f"⚠️  {w}")
        if not result["errors"]:
            print("\n✅ 没有格式错误")
        return

    if not args.snapshot:
        parser.error("需要 --snapshot / --check / --consistency 之一")

    rows = build_rows(SnapshotStore.load(args.snapshot))
    path = write_csv(rows, args.out)
    companies = {}
    for r in rows:
        companies[r["company"]] = companies.get(r["company"], 0) + 1
    print(f"已生成待标注文件：{path}")
    print(f"共 {len(rows)} 条，覆盖 {len(companies)} 家目标公司，"
          f"平均每家 {len(rows) / max(len(companies), 1):.0f} 条")
    print("每家条数：" + "、".join(f"{c} {n}" for c, n in sorted(companies.items())))
    print(f"\n只需填写 {FILLED_BY_ANNOTATOR} 三列，规范见 docs/annotation-guide.md")


if __name__ == "__main__":
    main()
