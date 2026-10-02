"""
eval/runner.py
---------------
批量执行实验：对每个（变体 × 目标公司 × 重复次数）生成一份报告，结果写成 JSON。

特点：
  - 全部数据来自快照回放，不访问外部接口；temperature 固定为 0
  - 断点续跑：已存在的结果文件默认跳过（--overwrite 可重跑）
  - 失败重试一次；仍失败则记为 status=failed 并保留错误信息，不中断整批
  - 记录 usage（调用次数 / token / 耗时）、stop_reason、代码与快照版本，便于复现

命令行：
  python -m eval.runner --variant A --smoke                 # 冒烟：2 家公司各 1 次
  python -m eval.runner --variant C --repeat 3              # 正式跑一个变体
  python -m eval.runner --variant all --repeat 3 --workers 2
  python -m eval.runner --summary                           # 查看已完成的结果统计
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import config
from eval import variants
from eval.snapshot import SnapshotStore, load_companies, seed_graph

DEFAULT_SNAPSHOT = "eval/snapshots/2026-09-19/snapshot.json"
DEFAULT_OUT = "eval/runs"
# tool_log 里体积大的内部字段不写进结果文件（指标计算用不到）
_DROP_TOOL_FIELDS = ("_graph", "_mermaid")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def slim_tool_log(tool_log: list) -> list:
    slim = []
    for entry in tool_log:
        result = entry.get("result") or {}
        slim.append({**entry, "result": {k: v for k, v in result.items() if k not in _DROP_TOOL_FIELDS}})
    return slim


def run_path(out_dir: str | Path, variant: str, company: str, repeat: int) -> Path:
    return Path(out_dir) / variant / f"{company}_r{repeat}.json"


def run_one(variant: str, company: str, repeat: int, store: SnapshotStore, graph,
            out_dir: str | Path = DEFAULT_OUT, build_agent: Optional[Callable] = None,
            retries: int = 1) -> dict:
    """跑一份报告并落盘，返回结果字典"""
    build_agent = build_agent or variants.build_agent
    path = run_path(out_dir, variant, company, repeat)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "variant": variant, "company": company, "repeat": repeat,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "snapshot_recorded_at": store.data.get("recorded_at"),
    }

    last_error = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            agent = build_agent(variant, store, company, graph=graph, temperature=0)
            session_id, report_md = agent.generate_report(company)
            draft, tool_log = agent.last_draft, agent.last_tool_log
            record.update({
                "status": "ok" if draft else "no_draft",
                "attempt": attempt + 1,
                "session_id": session_id,
                "draft": draft,
                "report_md": report_md,
                "tool_log": slim_tool_log(tool_log),
                "stop_reason": agent.loop.last_stop_reason,
                "compliance": {k: v for k, v in (agent.last_compliance or {}).items() if k != "draft"},
                "usage": agent.usage.totals(),
                "wall_seconds": round(time.perf_counter() - started, 2),
            })
            break
        except Exception as e:                     # noqa: BLE001 - 单次失败不应中断整批
            last_error = f"{type(e).__name__}: {e}"
            record.update({
                "status": "failed", "attempt": attempt + 1, "error": last_error,
                "traceback": traceback.format_exc()[-2000:],
                "wall_seconds": round(time.perf_counter() - started, 2),
            })
            if attempt < retries:
                time.sleep(2)

    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return record


def result_commit(path: Path) -> Optional[str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("git_commit")
    except Exception:                          # noqa: BLE001 - 读不出就当作旧结果
        return None


def plan_runs(variants_: list[str], companies: list[str], repeat: int,
              out_dir: str | Path, overwrite: bool,
              current_commit: Optional[str] = None, keep_stale: bool = False,
              on_notice: Callable[[str], None] = print) -> list[tuple[str, str, int]]:
    """列出待执行任务。

    断点续跑默认跳过已有结果，但**代码版本不同的旧结果会重跑**：一轮消融
    实验的所有结果必须来自同一份代码，否则变体之间的差异里混进了代码改动。
    （实测踩过：改完证据摘要后直接跑全量，前 6 份仍是旧版本的结果。）
    --keep-stale 可以关掉这个行为。
    """
    jobs, stale = [], []
    for variant in variants_:
        for company in companies:
            for r in range(1, repeat + 1):
                path = run_path(out_dir, variant, company, r)
                if overwrite or not path.exists():
                    jobs.append((variant, company, r))
                    continue
                commit = result_commit(path)
                if not keep_stale and current_commit and commit != current_commit:
                    stale.append((variant, company, r, commit))
                    jobs.append((variant, company, r))
    if stale:
        versions = sorted({c or "unknown" for *_, c in stale})
        on_notice(f"⚠️  {len(stale)} 份已有结果来自其它代码版本（{', '.join(versions)}，"
                  f"当前 {current_commit}），将重跑；加 --keep-stale 可保留它们")
    return jobs


def run_batch(jobs: list[tuple[str, str, int]], store: SnapshotStore, graph,
              out_dir: str | Path = DEFAULT_OUT, workers: int = 1,
              build_agent: Optional[Callable] = None,
              on_progress: Callable[[str], None] = print) -> list[dict]:
    total = len(jobs)
    done = 0

    def _run(job):
        nonlocal done
        variant, company, repeat = job
        record = run_one(variant, company, repeat, store, graph, out_dir, build_agent)
        done += 1
        usage = record.get("usage", {})
        on_progress(
            f"[{done}/{total}] {variant} {company} r{repeat} → {record['status']}"
            f"（{record.get('wall_seconds', 0):.1f}s，{usage.get('calls', 0)} 次调用，"
            f"{usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} tokens）"
            + (f"  ⚠️ {record.get('error')}" if record["status"] == "failed" else "")
        )
        return record

    if workers <= 1:
        return [_run(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, jobs))


def load_runs(out_dir: str | Path = DEFAULT_OUT) -> list[dict]:
    """读取结果文件；跳过 _reports/ 等辅助目录与非结果 JSON（如图谱快照）"""
    runs = []
    for path in sorted(Path(out_dir).glob("*/*.json")):
        if path.parent.name.startswith("_"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"跳过无法解析的文件：{path}")
            continue
        if isinstance(record, dict) and "variant" in record:
            runs.append(record)
    return runs


def summarize(runs: list[dict]) -> dict:
    by_variant: dict[str, dict] = {}
    for r in runs:
        agg = by_variant.setdefault(r["variant"], {
            "runs": 0, "ok": 0, "failed": 0, "no_draft": 0,
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "companies": set(),
        })
        agg["runs"] += 1
        agg[r["status"]] = agg.get(r["status"], 0) + 1
        agg["companies"].add(r["company"])
        usage = r.get("usage") or {}
        agg["calls"] += usage.get("calls", 0)
        agg["input_tokens"] += usage.get("input_tokens", 0)
        agg["output_tokens"] += usage.get("output_tokens", 0)
        agg["seconds"] += r.get("wall_seconds", 0)

    for agg in by_variant.values():
        n = max(agg["runs"], 1)
        agg["companies"] = len(agg["companies"])
        agg["avg_calls"] = round(agg["calls"] / n, 1)
        agg["avg_tokens"] = round((agg["input_tokens"] + agg["output_tokens"]) / n)
        agg["avg_seconds"] = round(agg["seconds"] / n, 1)
    return by_variant


def print_summary(by_variant: dict, on_progress: Callable[[str], None] = print):
    on_progress(f"\n{'变体':<6}{'份数':>6}{'成功':>6}{'失败':>6}{'无草稿':>8}"
                f"{'平均调用':>10}{'平均tokens':>12}{'平均耗时':>10}")
    for variant, agg in sorted(by_variant.items()):
        on_progress(f"{variant:<6}{agg['runs']:>6}{agg.get('ok', 0):>6}{agg.get('failed', 0):>6}"
                    f"{agg.get('no_draft', 0):>8}{agg['avg_calls']:>10}{agg['avg_tokens']:>12}"
                    f"{agg['avg_seconds']:>10}s")


# ── 命令行 ───────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="批量执行消融实验")
    parser.add_argument("--variant", default="C", help="A / B / C / all")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--companies", default="eval/companies.txt")
    parser.add_argument("--only", help="只跑这些公司，逗号分隔")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=1, help="并行度；注意 API 速率限制")
    parser.add_argument("--overwrite", action="store_true", help="重跑已有结果")
    parser.add_argument("--keep-stale", action="store_true",
                        help="保留其它代码版本跑出的已有结果（默认重跑，保证一轮实验同源）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要执行的任务")
    parser.add_argument("--smoke", action="store_true", help="冒烟：前 2 家公司各 1 次")
    parser.add_argument("--summary", action="store_true", help="统计已完成的结果")
    args = parser.parse_args(argv)

    if args.summary:
        runs = load_runs(args.out)
        if not runs:
            print(f"{args.out} 下没有结果文件")
            return
        print_summary(summarize(runs))
        failed = [r for r in runs if r["status"] != "ok"]
        if failed:
            print(f"\n非成功的 {len(failed)} 条：")
            for r in failed[:20]:
                print(f"  {r['variant']} {r['company']} r{r['repeat']}: {r['status']} "
                      f"{r.get('error', r.get('stop_reason', ''))}")
        return

    variant_list = list(variants.VARIANTS) if args.variant.lower() == "all" else [args.variant.upper()]
    companies = load_companies(args.companies)
    if args.only:
        wanted = {c.strip().upper() for c in args.only.split(",")}
        companies = [c for c in companies if c in wanted]
    repeat = args.repeat
    if args.smoke:
        companies, repeat = companies[:2], 1

    jobs = plan_runs(variant_list, companies, repeat, args.out, args.overwrite,
                     current_commit=git_commit(), keep_stale=args.keep_stale)
    print(f"待执行 {len(jobs)} 个任务"
          f"（变体 {', '.join(variant_list)}；公司 {len(companies)} 家；每个 {repeat} 次）")
    if args.dry_run or not jobs:
        for job in jobs[:30]:
            print(f"  {job[0]} {job[1]} r{job[2]}")
        if len(jobs) > 30:
            print(f"  ... 共 {len(jobs)} 个")
        return

    store = SnapshotStore.load(args.snapshot)
    graph = seed_graph()
    config.REPORTS_DIR = Path(args.out) / "_reports"      # 报告 Markdown 另存，避免混进 reports/
    started = time.perf_counter()
    run_batch(jobs, store, graph, args.out, args.workers)
    print(f"\n完成，总耗时 {(time.perf_counter() - started) / 60:.1f} 分钟")
    print_summary(summarize(load_runs(args.out)))


if __name__ == "__main__":
    main()
