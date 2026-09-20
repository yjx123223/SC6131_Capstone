"""
eval/judge.py
--------------
消融实验的评审模型判定。两件事：

1. **事件提及判定**（M1/M2 的基础）：给定一份报告和该公司的候选事件池，
   判断报告提到了哪些事件。表述不同但指向同一件事算提到，必须给出报告原句
   作为证据，便于人工复核。三个变体走**同一条判定路径**——不因为 C 组报告里
   有事件编号就改用字符串匹配，否则比较不公平。

2. **引用可核验判定**（M3/M4 的基础）：给定报告里的若干条论断和这份报告
   当时拿到的全部工具证据，逐条判断能不能在证据里找到依据。

设计约束：
  - temperature=0，且结果按 (模型, prompt) 的 sha256 落盘缓存。
    改判定提示词会换 key、自动重判；只是重跑脚本则直接命中缓存，不重复花钱。
  - 判定结果用强制 tool_use 输出，不解析自由文本 JSON。
  - 判定器本身要校验：--sample 导出若干条判定给人工复核，算一致率。
    见 docs/eval-design.md 5.3（kappa < 0.6 就要改提示词重来）。

命令行：
  python -m eval.judge --mentions              # 对 eval/runs 下所有结果做事件提及判定
  python -m eval.judge --citations             # 引用可核验判定
  python -m eval.judge --sample 50 --out eval/results/judge_sample.csv   # 导出人工复核样本
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Callable, Optional

import config
from eval import gold as gold_mod

DEFAULT_CACHE = "eval/results/judge_cache.json"
DEFAULT_RUNS = "eval/runs"
DEFAULT_GOLD = "eval/gold/gold_events.csv"


# ── 候选事件池 ───────────────────────────────────────────────────

def candidate_pool(gold_rows: list[dict], company: str, summary_chars: int = 220) -> list[dict]:
    """某目标公司的全部候选事件。

    **不过滤 material**：判定器不该知道哪些事件是"重要的"，否则等于把答案
    透给它。material / hop 的筛选在算指标时才做。
    """
    pool = []
    for r in gold_rows:
        if r.get("company") != company:
            continue
        pool.append({
            "id": int(r["id"]),
            "source_company": r.get("source_company", ""),
            "hop": int(r.get("hop") or 0),
            "relation": r.get("relation", ""),
            "published_at": r.get("published_at", ""),
            "title": r.get("title", ""),
            "summary": (r.get("summary") or "")[:summary_chars],
        })
    return pool


def _pool_text(pool: list[dict]) -> str:
    lines = []
    for e in pool:
        who = e["source_company"] if e["hop"] == 0 else f"{e['source_company']}（{e['relation']}）"
        lines.append(f"[{e['id']}] {e['published_at'][:10]} {who}：{e['title']}｜{e['summary']}")
    return "\n".join(lines)


MENTION_SYSTEM = """你是一位严谨的文本比对员。你会收到一份投资研究报告，以及一组候选事件。

判断报告中提到了哪些候选事件。判定标准：
- 表述不同但指向同一件事，算提到（例如事件写"Evercore 将目标价上调至 380 美元"，
  报告写"分析师给出 380 美元目标价"，算提到）。
- 只提到公司名、没有触及该事件的内容，不算提到。
- 报告里泛泛地说"供应链存在风险"，而候选事件是某家供应商的具体事件，不算提到。
- 证据必须从**报告正文**中逐字摘录，不得复制候选事件的标题或摘要，也不要自己改写。
  在报告里找不到可以逐字摘录的句子，就不要判为提到——宁可漏判，不要凑证据。

只做客观比对，不评价报告好坏。"""

MENTION_TOOL = {
    "name": "emit_mentions",
    "description": "输出报告提到的候选事件编号及证据原句",
    "input_schema": {
        "type": "object",
        "properties": {
            "mentioned": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "候选事件编号"},
                        "evidence": {"type": "string", "description": "报告中的原句，逐字摘录"},
                    },
                    "required": ["id", "evidence"],
                },
            },
        },
        "required": ["mentioned"],
    },
}


CITATION_SYSTEM = """你是一位严谨的事实核查员。你会收到若干条来自投资报告的论断，
以及生成这份报告时实际拿到的全部工具数据。

逐条判断该论断能否在工具数据中找到依据：
- supported：数值、事件、申报都能在数据中找到。单位换算（331839012864 写成 3318 亿）、
  四舍五入（34.04% 写成约 34%）、换一种说法，都算找得到。
- unsupported：出现了数据里没有的数值、公司、事件，或把数据里没有的因果说成事实。
- judgment：纯粹的主观判断或展望（如"估值偏高"、"建议关注"），不涉及可核对的事实。

必须为 supported 摘出数据中的对应片段，为 unsupported 说明缺了什么。"""

CITATION_TOOL = {
    "name": "emit_verdicts",
    "description": "逐条输出论断的核查结论",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "论断编号，从 1 开始"},
                        "verdict": {"type": "string", "enum": ["supported", "unsupported", "judgment"]},
                        "evidence": {"type": "string", "description": "数据中的对应片段，或缺失说明"},
                    },
                    "required": ["index", "verdict", "evidence"],
                },
            },
        },
        "required": ["verdicts"],
    },
}


# ── 论断抽取 ─────────────────────────────────────────────────────

_CLAUSE_SPLIT = re.compile(r"[（(]\d+[）)]")


def claims_of(draft: dict, max_risk_clauses: int = 6) -> list[str]:
    """报告里可核查的论断：关键信号逐条 + 风险提示按 (1)(2) 拆句"""
    claims = [s.strip() for s in (draft.get("key_signals") or []) if s and s.strip()]
    risk = (draft.get("risk_warnings") or "").strip()
    if risk:
        parts = [p.strip(" ；;。") for p in _CLAUSE_SPLIT.split(risk)]
        parts = [p for p in parts if len(p) > 8]
        claims += parts[:max_risk_clauses] if parts else [risk]
    return claims


# ── 模型返回的容错解析 ────────────────────────────────────────────
# Haiku 偶尔会把数组字段返回成"一个 JSON 字符串"，而且那串 JSON 里还可能有
# 未转义的引号（实测：evidence 里写了 具有"Apple DNA"）。直接迭代会把字符串
# 按字符拆开，json.loads 也会失败。这里做两级兜底：先 json.loads，
# 再正则抽取；两级都失败才报错。修复过的判定会标记出来，便于人工优先复核。

_MENTION_RE = re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"evidence"\s*:\s*"(.*?)"\s*[},]', re.S)
_VERDICT_RE = re.compile(
    r'"index"\s*:\s*(\d+)\s*,\s*"verdict"\s*:\s*"(\w+)"\s*,\s*"evidence"\s*:\s*"(.*?)"\s*[},]', re.S)
_ID_ONLY_RE = re.compile(r'"(?:id|index)"\s*:\s*(\d+)')


def _repair(text: str, kind: str) -> list[dict]:
    """从半结构化文本里抠出判定项。kind: "mention" | "verdict" """
    if kind == "mention":
        items = [{"id": int(i), "evidence": e.strip()} for i, e in _MENTION_RE.findall(text)]
    else:
        items = [{"index": int(i), "verdict": v, "evidence": e.strip()}
                 for i, v, e in _VERDICT_RE.findall(text)]
    if items:
        return items
    # 连字段顺序都对不上时，至少把编号救回来——编号决定指标，证据只影响人工复核
    ids = [int(x) for x in _ID_ONLY_RE.findall(text)]
    key = "id" if kind == "mention" else "index"
    return [{key: i, "evidence": "（模型返回格式异常，证据未能解析）",
             **({"verdict": "unparsed"} if kind == "verdict" else {})} for i in ids]


def _coerce_numeric(items: list[dict], kind: str) -> tuple[list[dict], int]:
    """编号字段统一成 int。模型有时返回 "3" 而不是 3：
    citations 里会让 1 <= index <= n 直接抛 TypeError；mentions 里更隐蔽——
    字符串不在整数集合里，会被当成"编造的编号"悄悄丢掉，指标无声偏低。"""
    key = "id" if kind == "mention" else "index"
    out, fixed = [], 0
    for item in items:
        value = item.get(key)
        if isinstance(value, str):
            digits = value.strip().lstrip("Ee#[](（）) ").rstrip("] )）")
            if digits.isdigit():
                item = {**item, key: int(digits)}
                fixed += 1
            else:
                continue                       # 编号无法解析，整条丢弃
        elif not isinstance(value, int):
            continue
        out.append(item)
    return out, fixed


def coerce_items(value, kind: str) -> tuple[list[dict], str]:
    """把模型返回的字段规范成 list[dict]，并返回修复说明（空串表示原本就正常）"""
    notes = []
    if value is None:
        return [], "字段缺失"
    if isinstance(value, dict):
        items, notes = [value], ["返回了单个对象而非数组"]
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            items = _repair(value, kind)
            notes = [f"数组被返回成字符串且 JSON 非法，正则抽出 {len(items)} 项"]
        else:
            items = parsed if isinstance(parsed, list) else [parsed]
            items = [x for x in items if isinstance(x, dict)]
            notes = ["数组被返回成字符串，已解析"]
    elif isinstance(value, list):
        items = [x for x in value if isinstance(x, dict)]
        if len(items) != len(value):
            notes = [f"丢弃了 {len(value) - len(items)} 个非对象项"]
    else:
        return [], f"无法解析的类型 {type(value).__name__}"

    before = len(items)
    items, fixed = _coerce_numeric(items, kind)
    if fixed:
        notes.append(f"{fixed} 个编号是字符串，已转成整数")
    if len(items) != before:
        notes.append(f"丢弃了 {before - len(items)} 条编号无法解析的判定")
    return items, "；".join(notes)


# ── 判定器 ───────────────────────────────────────────────────────

class Judge:
    """评审模型调用 + 落盘缓存。client 传 None 时只能命中缓存（离线跑测试用）"""

    def __init__(self, client=None, model: str = None, max_tokens: int = None,
                 temperature=None, cache_path: str | Path = DEFAULT_CACHE):
        self.client = client
        self.model = model or config.JUDGE_MODEL
        self.max_tokens = max_tokens or config.JUDGE_MAX_TOKENS
        self.temperature = config.JUDGE_TEMPERATURE if temperature is None else temperature
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) \
            if self.cache_path.exists() else {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    # -- 缓存 --
    def _key(self, tool_name: str, system: str, user: str) -> str:
        blob = json.dumps([self.model, self.temperature, tool_name, system, user],
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.cache_path)

    def _call(self, tool: dict, system: str, user: str) -> dict:
        key = self._key(tool["name"], system, user)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
        if self.client is None:
            raise RuntimeError("判定结果不在缓存里，且没有传入 client（离线模式）")

        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      system=system, messages=[{"role": "user", "content": user}],
                      tools=[tool], tool_choice={"type": "tool", "name": tool["name"]})
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self.client.messages.create(**kwargs)
        result = next((b.input for b in response.content if getattr(b, "type", "") == "tool_use"), None)
        if result is None:
            raise RuntimeError(f"评审模型未返回 {tool['name']}（stop_reason={getattr(response, 'stop_reason', '?')}）")

        with self._lock:
            self.cache[key] = result
            self.misses += 1
            self.save()
        return result

    # -- 两类判定 --
    def mentions(self, report_md: str, pool: list[dict]) -> dict:
        user = (f"# 候选事件\n{_pool_text(pool)}\n\n"
                f"# 投资研究报告\n{report_md}")
        result = self._call(MENTION_TOOL, MENTION_SYSTEM, user)
        items, repaired = coerce_items(result.get("mentioned"), "mention")
        valid = {e["id"] for e in pool}
        # 模型可能编出不存在的编号，丢掉并记录
        kept = [m for m in items if m.get("id") in valid]
        out = {"mentioned": kept, "dropped": [m for m in items if m.get("id") not in valid]}
        if repaired:
            out["repaired"] = repaired
        return out

    def citations(self, claims: list[str], evidence_text: str) -> dict:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        user = (f"# 待核查的论断\n{numbered}\n\n"
                f"# 生成报告时拿到的全部工具数据\n{evidence_text}")
        result = self._call(CITATION_TOOL, CITATION_SYSTEM, user)
        items, repaired = coerce_items(result.get("verdicts"), "verdict")
        verdicts = [v for v in items if 1 <= v.get("index", 0) <= len(claims)]
        out = {"verdicts": verdicts, "claims": claims}
        if repaired:
            out["repaired"] = repaired
        return out


# ── 证据核验 ────────────────────────────────────────────────────
# 判定器偶尔会把候选事件的正文当成证据交上来（实测 26 条里有 2 条）。
# 那是假阳性，会虚高召回率。这里做确定性核验：证据必须能在报告正文里找到。
# 比对前规范化掉空白、中英文引号与全角标点——报告写 "买入区域"、
# 判定器写 '买入区域'，是同一句话，不该算不匹配。

_QUOTE_CHARS = "\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f'\"`"
_NORMALIZE = str.maketrans({c: "" for c in _QUOTE_CHARS}
                           | {"（": "(", "）": ")", "，": ",", "：": ":", "、": ",", "％": "%"})


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").translate(_NORMALIZE)).lower()


def evidence_verified(evidence: str, report_md: str, probe_chars: int = 24) -> bool:
    """证据的前 probe_chars 个规范化字符能否在报告里找到"""
    probe = normalize_for_match(evidence)[:probe_chars]
    return bool(probe) and probe in normalize_for_match(report_md)


def verify_mentions(mentions: list[dict], report_md: str) -> list[dict]:
    """给每条判定打上 verified 标记（不删除，便于人工复核假阳性）"""
    return [{**m, "verified": evidence_verified(m.get("evidence", ""), report_md)} for m in mentions]


# ── 批量判定 ─────────────────────────────────────────────────────

DEFAULT_JUDGEMENTS = "eval/results/judgements"


def judgement_path(out_dir: str | Path, variant: str, company: str, repeat: int) -> Path:
    return Path(out_dir) / variant / f"{company}_r{repeat}.json"


def judge_run(judge: Judge, run: dict, pool: list[dict], evidence_text: str,
              do_mentions: bool = True, do_citations: bool = True) -> dict:
    out = {"variant": run["variant"], "company": run["company"], "repeat": run["repeat"],
           "run_git_commit": run.get("git_commit"), "judge_model": judge.model}
    draft = run.get("draft") or {}
    if do_mentions:
        report_md = run.get("report_md") or ""
        mentions = judge.mentions(report_md, pool)
        mentions["mentioned"] = verify_mentions(mentions["mentioned"], report_md)
        out["mentions"] = mentions
    if do_citations:
        claims = claims_of(draft)
        out["citations"] = judge.citations(claims, evidence_text) if claims else {"verdicts": [], "claims": []}
    return out


def judge_all(runs: list[dict], gold_rows: list[dict], judge: Judge,
              out_dir: str | Path = DEFAULT_JUDGEMENTS,
              do_mentions: bool = True, do_citations: bool = True,
              on_progress: Callable[[str], None] = print) -> list[dict]:
    from tool_log_summary import summarize_tool_log

    pools = {}
    results, failures = [], []
    for i, run in enumerate(runs, 1):
        company = run["company"]
        pools.setdefault(company, candidate_pool(gold_rows, company))
        evidence = summarize_tool_log(run.get("tool_log") or [])
        try:
            result = judge_run(judge, run, pools[company], evidence, do_mentions, do_citations)
        except Exception as e:                # noqa: BLE001 - 单份失败不该中断整批
            failures.append((run["variant"], company, run["repeat"], f"{type(e).__name__}: {e}"))
            on_progress(f"[{i}/{len(runs)}] {run['variant']} {company} r{run['repeat']}："
                        f"⚠️ 判定失败 {type(e).__name__}: {e}")
            continue
        path = judgement_path(out_dir, run["variant"], company, run["repeat"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        mentioned = result.get("mentions", {}).get("mentioned", [])
        n_m = len(mentioned)
        n_unverified = sum(1 for m in mentioned if not m.get("verified", True))
        n_c = result.get("citations", {}).get("verdicts", [])
        unsup = sum(1 for v in n_c if v["verdict"] == "unsupported")
        repaired = [v.get("repaired") for v in (result.get("mentions"), result.get("citations"))
                    if isinstance(v, dict) and v.get("repaired")]
        on_progress(f"[{i}/{len(runs)}] {run['variant']} {company} r{run['repeat']}："
                    f"提及 {n_m} 个事件"
                    + (f"（{n_unverified} 条证据未核实）" if n_unverified else "")
                    + f"，论断 {len(n_c)} 条（无依据 {unsup}）"
                    f"｜缓存命中 {judge.hits} / 新判 {judge.misses}"
                    + (f"  🔧 {'; '.join(repaired)}" if repaired else ""))
    if failures:
        on_progress(f"\n⚠️ {len(failures)} 份判定失败，重跑同一条命令会补上：")
        for f in failures[:10]:
            on_progress(f"   {f[0]} {f[1]} r{f[2]}: {f[3]}")
    return results


def load_judgements(out_dir: str | Path = DEFAULT_JUDGEMENTS) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(out_dir).glob("*/*.json"))]


def sample_for_review(judgements: list[dict], gold_rows: list[dict], n: int = 50,
                      seed: int = 20260920) -> list[dict]:
    """抽一批判定给人工复核，用来算判定器与人工的一致率（design 5.3）"""
    import random
    by_id = {int(r["id"]): r for r in gold_rows}
    rows = []
    for j in judgements:
        for m in j.get("mentions", {}).get("mentioned", []):
            event = by_id.get(m["id"], {})
            rows.append({
                "kind": "mention", "variant": j["variant"], "company": j["company"],
                "repeat": j["repeat"], "event_id": m["id"],
                "event": f"{event.get('source_company', '')}｜{event.get('title', '')}",
                "judge_verdict": "提到", "evidence": m.get("evidence", ""),
                "human_verdict": "", "note": "",
            })
        for v in j.get("citations", {}).get("verdicts", []):
            claims = j["citations"].get("claims", [])
            idx = v["index"] - 1
            rows.append({
                "kind": "citation", "variant": j["variant"], "company": j["company"],
                "repeat": j["repeat"], "event_id": "",
                "event": claims[idx] if 0 <= idx < len(claims) else "",
                "judge_verdict": v["verdict"], "evidence": v.get("evidence", ""),
                "human_verdict": "", "note": "",
            })
    random.Random(seed).shuffle(rows)
    return rows[:n]


def write_sample_csv(rows: list[dict], path: str | Path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["kind", "variant", "company", "repeat", "event_id", "event",
                                 "judge_verdict", "evidence", "human_verdict", "note"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    from eval.runner import load_runs

    parser = argparse.ArgumentParser(description="评审模型判定（事件提及 / 引用可核验）")
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--out", default=DEFAULT_JUDGEMENTS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--mentions", action="store_true", help="只做事件提及判定")
    parser.add_argument("--citations", action="store_true", help="只做引用可核验判定")
    parser.add_argument("--only", help="只判定这些公司，逗号分隔")
    parser.add_argument("--variant", help="只判定某个变体")
    parser.add_argument("--limit", type=int, help="只判定前 N 份（试跑用）")
    parser.add_argument("--sample", type=int, help="从已有判定里抽 N 条导出人工复核")
    parser.add_argument("--sample-out", default="eval/results/judge_sample.csv")
    args = parser.parse_args(argv)

    gold_rows = gold_mod.read_csv(args.gold)

    if args.sample:
        rows = sample_for_review(load_judgements(args.out), gold_rows, args.sample)
        path = write_sample_csv(rows, args.sample_out)
        print(f"已导出 {len(rows)} 条判定到 {path}，填 human_verdict 列后用 --review 算一致率")
        return

    runs = load_runs(args.runs)
    if args.variant:
        runs = [r for r in runs if r["variant"] == args.variant.upper()]
    if args.only:
        wanted = {c.strip().upper() for c in args.only.split(",")}
        runs = [r for r in runs if r["company"] in wanted]
    runs = sorted(runs, key=lambda r: (r["variant"], r["company"], r["repeat"]))
    if args.limit:
        runs = runs[:args.limit]
    if not runs:
        print("没有待判定的结果")
        return

    do_m = args.mentions or not args.citations
    do_c = args.citations or not args.mentions

    import anthropic
    client = anthropic.Anthropic(api_key=config.get_anthropic_api_key())
    judge = Judge(client, cache_path=args.cache)
    print(f"待判定 {len(runs)} 份（事件提及={do_m}，引用核验={do_c}），"
          f"缓存 {len(judge.cache)} 条")
    judge_all(runs, gold_rows, judge, args.out, do_m, do_c)
    print(f"\n完成：缓存命中 {judge.hits}，新判定 {judge.misses}")


if __name__ == "__main__":
    main()
