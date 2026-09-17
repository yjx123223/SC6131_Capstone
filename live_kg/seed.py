"""
live_kg/seed.py
----------------
读取人工种子关系文件（CSV：subject,relation,object,note）。

种子文件两端直接写 ticker，不需要实体链接；读取时逐行校验：
关系必须属于 ontology.SEED_RELATIONS，ticker 格式合法，不能自环。
非法行不会中断读取，而是收集到 errors 里返回，方便发现手工录入错误。
"""

import csv
import re
from pathlib import Path
from typing import Optional

import config
from .ontology import SEED_RELATIONS

_TICKER = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def load_seed_relations(path: Optional[str | Path] = None) -> tuple[list[dict], list[str]]:
    """
    Returns
    -------
    (rows, errors)
      rows   : [{"subject", "relation", "object", "note", "line"}]，已去重
      errors : ["第 N 行：原因", ...]
    """
    path = Path(path or config.KG_SEED_PATH)
    if not path.exists():
        return [], [f"种子文件不存在：{path}"]

    rows, errors, seen = [], [], set()
    with path.open(encoding="utf-8") as f:
        lines = [(i, l) for i, l in enumerate(f, 1) if l.strip() and not l.lstrip().startswith("#")]

    reader = csv.DictReader((l for _, l in lines))
    for (line_no, _), rec in zip(lines[1:], reader):
        subj = (rec.get("subject") or "").strip().upper()
        rel = (rec.get("relation") or "").strip().upper()
        obj = (rec.get("object") or "").strip().upper()
        note = (rec.get("note") or "").strip()

        if rel not in SEED_RELATIONS:
            errors.append(f"第 {line_no} 行：关系 {rel!r} 不允许出现在种子文件中")
            continue
        if not _TICKER.match(subj) or not _TICKER.match(obj):
            errors.append(f"第 {line_no} 行：ticker 格式不合法（{subj!r}, {obj!r}）")
            continue
        if subj == obj:
            errors.append(f"第 {line_no} 行：不能自环（{subj}）")
            continue

        key = (subj, rel, obj)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"subject": subj, "relation": rel, "object": obj, "note": note, "line": line_no})

    return rows, errors
