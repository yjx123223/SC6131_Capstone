"""
eval/gold.py
-------------
从快照生成人工标注用的候选事件池，并对标注结果做校验与统计。

候选池口径（与各变体可见的数据一致）：
  目标公司自己的新闻      最多 config.NEWS_MAX_ITEMS 条（hop=0）
  每个 1~2 跳邻居的新闻   最多 config.KG_NEWS_PER_COMPANY 条（hop=1/2）

标注规范见 docs/annotation-guide.md。标注者只填 material / impact_on_target / note 三列。

AI 起草（需人工审核）：逐篇文章的判断写在 eval/gold/ai_article_labels.json，
再由 derive_row_label() 按每行的关系推导出 material / impact_on_target，
保证同一篇文章在不同目标公司下的判断一致。

命令行：
  python -m eval.gold --snapshot eval/snapshots/2026-09-19/snapshot.json    # 生成待标注 CSV
  python -m eval.gold --apply-draft --snapshot <快照>                        # 写入 AI 草稿
  python -m eval.gold --check eval/gold/gold_events.csv                     # 校验 + 统计
  python -m eval.gold --consistency 第一次.csv 第二次.csv                    # 自一致率
  python -m eval.gold --review-diff 草稿.csv 审核后.csv                      # 人工修正率
"""

from __future__ import annotations

import argparse
import csv
import json
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


# ── AI 草稿：逐篇文章判断 → 逐行标注 ──────────────────────────────

DRAFT_PREFIX = "AI草稿："
_INVERT = {"positive": "negative", "negative": "positive", "neutral": "neutral"}

# 对目标公司而言，邻居事件的方向如何映射
#   供应商/客户/合作伙伴：同向（供应商利空 → 对目标利空）
#   竞争对手：反向（对手利好 → 对目标是竞争压力）
_SAME_DIRECTION_ROLES = {"供应商", "客户", "合作伙伴", "上游供应商"}


def load_article_labels(path: str | Path = "eval/gold/ai_article_labels.json") -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def article_keys(store) -> dict:
    """url → "TICKER-序号"（序号按该公司新闻的发布时间倒序，与 build_rows 一致）"""
    keys = {}
    for ticker in store.universe:
        articles = sorted(store.articles(ticker), key=lambda a: a.get("published_at", ""), reverse=True)
        for i, a in enumerate(articles):
            keys.setdefault((ticker, a.get("url", "")), f"{ticker}-{i}")
    return keys


def derive_row_label(row: dict, label: dict, seen_urls: Optional[set] = None) -> tuple[str, str, str]:
    """
    把一篇文章的判断映射到某一行（某个目标公司视角）。

    Returns (material, impact_on_target, note)
    """
    note = label.get("note", "")
    if not label.get("event"):
        return "0", "", note

    # 同一目标公司下重复出现的同一篇文章（可能来自不同来源公司）只算一次
    if seen_urls is not None and row.get("url") in seen_urls:
        return "0", "", "与前一条重复（同一篇报道）"
    if label.get("dup_of"):
        return "0", "", note

    polarity = label.get("polarity", "neutral")
    about = label.get("about")
    relation = row.get("relation", "")
    hop = str(row.get("hop", "0"))

    # 报道主角就是目标公司本身 → 直接采用该方向
    if about == row["company"] or (not about and hop == "0"):
        return "1", polarity, note
    # 主角是别的公司，且不是目标 → 与目标无关
    if about and about != row["company"]:
        return "0", "", (note + "；对本目标无直接关系").lstrip("；")

    if not label.get("transmits"):
        return "0", "", (note + "；对目标公司的传导通路不明确").lstrip("；")

    downstream = label.get("downstream")
    if downstream:
        impact = downstream
    elif label.get("industry_wide"):
        impact = polarity
    elif any(r in relation for r in _SAME_DIRECTION_ROLES):
        impact = polarity
    else:                      # 竞争对手：方向取反
        impact = _INVERT.get(polarity, "neutral")
    return "1", impact, note


def apply_draft(rows: list[dict], store, labels: Optional[dict] = None,
                only_empty: bool = True) -> tuple[list[dict], dict]:
    """把 AI 草稿写入行（默认只填空白单元格），返回 (rows, 统计)"""
    labels = labels or load_article_labels()
    keys = article_keys(store)
    stats = {"filled": 0, "material": 0, "skipped_existing": 0, "missing_label": 0}
    seen: dict[str, set] = {}

    for row in rows:
        if only_empty and (row.get("material") or "").strip():
            stats["skipped_existing"] += 1
            continue
        key = keys.get((row["source_company"], row["url"]))
        label = labels.get(key) if key else None
        if label is None:
            stats["missing_label"] += 1
            continue

        seen_urls = seen.setdefault(row["company"], set())
        material, impact, note = derive_row_label(row, label, seen_urls)
        if material == "1":
            seen_urls.add(row["url"])
            stats["material"] += 1
        row["material"] = material
        row["impact_on_target"] = impact
        row["note"] = f"{DRAFT_PREFIX}{note}" if note else DRAFT_PREFIX
        stats["filled"] += 1
    return rows, stats


def review_diff(draft: list[dict], reviewed: list[dict]) -> dict:
    """人工审核修正率：草稿 vs 审核后"""
    draft_map = {r["id"]: r for r in draft}
    changed_material, changed_impact, changes = 0, 0, []
    compared = 0
    for r in reviewed:
        d = draft_map.get(r["id"])
        if d is None:
            continue
        compared += 1
        dm, rm = (d.get("material") or "").strip(), (r.get("material") or "").strip()
        di = (d.get("impact_on_target") or "").strip().lower()
        ri = (r.get("impact_on_target") or "").strip().lower()
        if dm != rm:
            changed_material += 1
        if di != ri:
            changed_impact += 1
        if dm != rm or di != ri:
            changes.append({"id": r["id"], "company": r.get("company"),
                            "draft": (dm, di), "reviewed": (rm, ri),
                            "title": (r.get("title") or "")[:60]})
    return {
        "compared": compared,
        "changed_material": changed_material,
        "changed_impact": changed_impact,
        "material_change_rate": round(changed_material / compared, 3) if compared else None,
        "any_change_rate": round(len(changes) / compared, 3) if compared else None,
        "changes": changes,
    }


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
    parser.add_argument("--apply-draft", action="store_true", help="用 AI 草稿填充（需要 --snapshot）")
    parser.add_argument("--labels", default="eval/gold/ai_article_labels.json")
    parser.add_argument("--review-diff", nargs=2, metavar=("DRAFT", "REVIEWED"), help="统计人工修正率")
    args = parser.parse_args(argv)

    if args.consistency:
        result = consistency(read_csv(args.consistency[0]), read_csv(args.consistency[1]))
        print(f"比较 {result['compared']} 条：material 一致率 {result['material_agreement']}，"
              f"含方向的完全一致率 {result['full_agreement']}")
        for d in result["disagreements"]:
            print(f"  #{d['id']}: {d['first']} → {d['second']}")
        return

    if args.review_diff:
        result = review_diff(read_csv(args.review_diff[0]), read_csv(args.review_diff[1]))
        print(f"对比 {result['compared']} 行：material 改动 {result['changed_material']} 行"
              f"（{result['material_change_rate']:.1%}），方向改动 {result['changed_impact']} 行；"
              f"总改动率 {result['any_change_rate']:.1%}")
        for c in result["changes"][:50]:
            print(f"  #{c['id']} [{c['company']}] {c['draft']} → {c['reviewed']}  {c['title']}")
        return

    if args.apply_draft:
        if not args.snapshot:
            parser.error("--apply-draft 需要 --snapshot")
        store = SnapshotStore.load(args.snapshot)
        rows = read_csv(args.out) if Path(args.out).exists() else build_rows(store)
        rows, stats = apply_draft(rows, store, load_article_labels(args.labels))
        write_csv(rows, args.out)
        print(f"AI 草稿已写入 {args.out}")
        print(f"  填充 {stats['filled']} 行，其中 material=1 共 {stats['material']} 行；"
              f"跳过已填 {stats['skipped_existing']} 行，找不到文章判断 {stats['missing_label']} 行")
        print("\n请逐条审核（重点看所有 material=1 的行），改完后运行：")
        print(f"  python -m eval.gold --check {args.out}")
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
