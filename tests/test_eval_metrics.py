"""eval/metrics.py 的单元测试：全部用构造数据，不依赖真实跑批结果"""
import json

import pytest

from eval import metrics


GOLD = [
    # AAPL：material 的跨公司事件 2 个（其中 1 个是 2 跳），本公司事件 1 个
    {"id": "1", "company": "AAPL", "hop": "0", "material": "1"},
    {"id": "2", "company": "AAPL", "hop": "1", "material": "1"},
    {"id": "3", "company": "AAPL", "hop": "2", "material": "1"},
    {"id": "4", "company": "AAPL", "hop": "1", "material": "0"},   # 非 material 不计入
    # MSFT：只有本公司事件，没有跨公司 material 事件 → M1 分母为 0
    {"id": "5", "company": "MSFT", "hop": "0", "material": "1"},
]


def test_gold_slices_only_counts_material():
    sl = metrics.gold_slices(GOLD)
    assert sl["AAPL"]["cross"] == {2, 3}
    assert sl["AAPL"]["hop2"] == {3}
    assert sl["AAPL"]["hop0"] == {1}
    assert sl["MSFT"]["cross"] == set()


def test_recall_returns_none_for_empty_denominator():
    """没有这类事件 ≠ 召回率 0 分"""
    assert metrics.recall({1}, set()) is None
    assert metrics.recall(set(), {1, 2}) == 0.0
    assert metrics.recall({2, 3}, {2, 3}) == 1.0
    assert metrics.recall({2}, {2, 3}) == 0.5


# ── 单份报告 ────────────────────────────────────────────────────

def _run(variant, company, repeat, rec="持有", usage=None, tool_log=None, draft_extra=None):
    draft = {"recommendation": rec, "confidence": "medium", "key_signals": ["s"]}
    draft.update(draft_extra or {})
    return {"variant": variant, "company": company, "repeat": repeat, "draft": draft,
            "wall_seconds": 50.0, "tool_log": tool_log or [],
            "usage": usage or {"calls": 4, "input_tokens": 20000, "output_tokens": 4000}}


def _judgement(variant, company, repeat, ids, verdicts=("supported", "unsupported", "judgment")):
    return {"variant": variant, "company": company, "repeat": repeat,
            "mentions": {"mentioned": [{"id": i, "evidence": "e"} for i in ids]},
            "citations": {"claims": list(verdicts),
                          "verdicts": [{"index": i, "verdict": v, "evidence": "x"}
                                       for i, v in enumerate(verdicts, 1)]}}


def test_run_metrics_computes_recalls_and_citation_rates():
    row = metrics.run_metrics(_run("C", "AAPL", 1), _judgement("C", "AAPL", 1, [1, 2]),
                              metrics.gold_slices(GOLD))
    assert row["m1_cross_recall"] == 0.5            # 命中 2，漏掉 3
    assert row["m2_hop2_recall"] == 0.0
    assert row["m2b_hop0_recall"] == 1.0
    assert (row["cross_hit"], row["cross_total"]) == (1, 2)
    # supported / unsupported 计入分母，judgment 不计入
    assert row["m3_citation_supported"] == 0.5
    assert row["m4_unsupported"] == 0.5
    assert row["claims_checkable"] == 2 and row["claims_judgment"] == 1


def test_run_metrics_none_when_no_cross_events():
    row = metrics.run_metrics(_run("A", "MSFT", 1), _judgement("A", "MSFT", 1, [5]),
                              metrics.gold_slices(GOLD))
    assert row["m1_cross_recall"] is None
    assert row["m2b_hop0_recall"] == 1.0


def test_fabricated_event_ids_detected():
    """报告引用了图谱返回里没有的事件编号"""
    tool_log = [{"tool": "query_company_graph", "result": {
        "target_events": [{"id": "E1"}], "propagated_risks": [{"event_id": "E2"}]}}]
    row = metrics.run_metrics(
        _run("C", "AAPL", 1, tool_log=tool_log, draft_extra={"cited_event_ids": ["E1", "E9"]}),
        _judgement("C", "AAPL", 1, []), metrics.gold_slices(GOLD))
    assert row["fabricated_event_ids"] == ["E9"]


def test_no_fabrication_when_nothing_cited():
    row = metrics.run_metrics(_run("A", "AAPL", 1), _judgement("A", "AAPL", 1, []),
                              metrics.gold_slices(GOLD))
    assert row["fabricated_event_ids"] == []


def test_build_rows_rejects_unjudged_runs():
    with pytest.raises(SystemExit, match="还没判定"):
        metrics.build_rows([_run("A", "AAPL", 1)], [], GOLD)


# ── M6 稳定性 ───────────────────────────────────────────────────

def test_jaccard():
    assert metrics.jaccard(set(), set()) == 1.0
    assert metrics.jaccard({1, 2}, {1, 2}) == 1.0
    assert metrics.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert metrics.jaccard({1}, set()) == 0.0


def test_stability_over_three_repeats():
    sl = metrics.gold_slices(GOLD)
    rows = [metrics.run_metrics(_run("C", "AAPL", r, rec=rec), _judgement("C", "AAPL", r, ids), sl)
            for r, rec, ids in [(1, "持有", [2]), (2, "持有", [2]), (3, "增持", [2, 3])]]
    s = metrics.stability(rows)[("C", "AAPL")]
    assert s["n"] == 3
    assert s["rec_identical"] is False
    assert s["rec_majority"] == pytest.approx(2 / 3)
    # 三对：{2}vs{2}=1，{2}vs{2,3}=0.5，{2}vs{2,3}=0.5
    assert s["mention_jaccard"] == pytest.approx((1 + 0.5 + 0.5) / 3)


def test_stability_identical_runs():
    sl = metrics.gold_slices(GOLD)
    rows = [metrics.run_metrics(_run("A", "AAPL", r), _judgement("A", "AAPL", r, [2]), sl)
            for r in (1, 2, 3)]
    s = metrics.stability(rows)[("A", "AAPL")]
    assert s["rec_identical"] is True and s["mention_jaccard"] == 1.0


# ── 汇总与配对 ───────────────────────────────────────────────────

def _dataset():
    sl = metrics.gold_slices(GOLD)
    rows = []
    plan = {"A": [2], "B": [2], "C": [2, 3]}          # C 多召回 2 跳事件
    for v, ids in plan.items():
        for r in (1, 2, 3):
            rows.append(metrics.run_metrics(_run(v, "AAPL", r), _judgement(v, "AAPL", r, ids), sl))
            rows.append(metrics.run_metrics(_run(v, "MSFT", r), _judgement(v, "MSFT", r, [5]), sl))
    return rows


def test_by_company_takes_median_over_repeats():
    per = metrics.by_company(_dataset())
    assert per[("C", "AAPL")]["m1_cross_recall"] == 1.0
    assert per[("A", "AAPL")]["m1_cross_recall"] == 0.5
    assert per[("A", "MSFT")]["m1_cross_recall"] is None


def test_paired_companies_excludes_company_missing_in_any_variant():
    per = metrics.by_company(_dataset())
    assert metrics.paired_companies(per, "m1_cross_recall", ["A", "B", "C"]) == ["AAPL"]


def test_summarize_uses_only_paired_companies():
    summary = metrics.summarize(_dataset())
    assert summary["m1_paired_companies"] == ["AAPL"]
    assert summary["m1_excluded"] == ["MSFT"]          # M1 分母为 0，三组一起排除
    assert summary["by_variant"]["C"]["m1_mean"] == 1.0
    assert summary["by_variant"]["A"]["m1_mean"] == 0.5
    assert summary["by_variant"]["A"]["m6_rec_identical"] == 1.0
    assert summary["by_variant"]["C"]["m5_calls"] == 4.0


def test_print_summary_mentions_exclusion(capsys):
    metrics.print_summary(metrics.summarize(_dataset()))
    out = capsys.readouterr().out
    assert "MSFT" in out and "配对公司 1 家" in out
    assert "M1跨公司召回" in out


def test_write_csv_flattens_list_fields(tmp_path):
    rows = _dataset()
    path = metrics.write_csv(rows, tmp_path / "metrics.csv")
    from eval.gold import read_csv
    back = read_csv(path)
    assert len(back) == len(rows)
    assert "mentioned_ids" in back[0]
    assert " " in back[0]["mentioned_ids"] or back[0]["mentioned_ids"].isdigit() or back[0]["mentioned_ids"] == ""


def test_main_end_to_end(tmp_path, capsys, monkeypatch):
    runs_dir, jud_dir = tmp_path / "runs", tmp_path / "jud"
    sl_rows = _dataset()
    for v in ("A", "B", "C"):
        for c in ("AAPL", "MSFT"):
            for r in (1, 2, 3):
                p = runs_dir / v / f"{c}_r{r}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(_run(v, c, r)), encoding="utf-8")
                q = jud_dir / v / f"{c}_r{r}.json"
                q.parent.mkdir(parents=True, exist_ok=True)
                ids = [2, 3] if (v == "C" and c == "AAPL") else ([2] if c == "AAPL" else [5])
                q.write_text(json.dumps(_judgement(v, c, r, ids)), encoding="utf-8")

    gold_csv = tmp_path / "gold.csv"
    import csv as _csv
    with gold_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "company", "hop", "material"])
        w.writeheader(); w.writerows(GOLD)

    metrics.main(["--runs", str(runs_dir), "--judgements", str(jud_dir),
                  "--gold", str(gold_csv), "--out", str(tmp_path / "results"), "--by-company"])
    out = capsys.readouterr().out
    assert "M1跨公司召回" in out and "未计入" in out
    assert (tmp_path / "results" / "metrics.csv").exists()
    summary = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert summary["by_variant"]["C"]["m1_mean"] == 1.0


# ── 证据核验：判定器摘了候选事件原文的那些提及不算数 ──────────────────

def _judgement_with_verification(ids_verified, ids_unverified):
    return {"variant": "C", "company": "AAPL", "repeat": 1,
            "mentions": {"mentioned":
                         [{"id": i, "evidence": "报告原句", "verified": True} for i in ids_verified]
                         + [{"id": i, "evidence": "候选事件原文", "verified": False} for i in ids_unverified]},
            "citations": {"claims": [], "verdicts": []}}


def test_unverified_mentions_excluded_by_default():
    sl = metrics.gold_slices(GOLD)
    row = metrics.run_metrics(_run("C", "AAPL", 1), _judgement_with_verification([2], [3]), sl)
    assert row["m1_cross_recall"] == 0.5            # 只算核实过的事件 2
    assert row["mentioned_ids"] == [2]
    assert row["mentions_unverified"] == 1


def test_count_unverified_flag_includes_them():
    sl = metrics.gold_slices(GOLD)
    row = metrics.run_metrics(_run("C", "AAPL", 1), _judgement_with_verification([2], [3]),
                              sl, count_unverified=True)
    assert row["m1_cross_recall"] == 1.0
    assert row["mentions_unverified"] == 1          # 仍然如实记录条数


def test_missing_verified_field_treated_as_verified():
    """旧判定文件没有 verified 字段时不能全判成假阳性"""
    sl = metrics.gold_slices(GOLD)
    j = {"variant": "A", "company": "AAPL", "repeat": 1,
         "mentions": {"mentioned": [{"id": 2, "evidence": "x"}]},
         "citations": {"claims": [], "verdicts": []}}
    row = metrics.run_metrics(_run("A", "AAPL", 1), j, sl)
    assert row["m1_cross_recall"] == 0.5 and row["mentions_unverified"] == 0


def test_summary_reports_unverified_count(capsys):
    sl = metrics.gold_slices(GOLD)
    rows = [metrics.run_metrics(_run(v, "AAPL", r), _judgement_with_verification([2], [3] if v == "C" else []), sl)
            for v in ("A", "B", "C") for r in (1, 2, 3)]
    metrics.print_summary(metrics.summarize(rows))
    out = capsys.readouterr().out
    assert "证据未能在报告里核实" in out and "'C': 3" in out
