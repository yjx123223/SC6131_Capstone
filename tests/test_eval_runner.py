"""eval/runner.py 的单元测试：全部用假 agent，不访问网络与 API"""
import json
import types

import pytest

from eval import runner


class FakeUsage:
    def __init__(self, totals):
        self._totals = totals

    def totals(self):
        return dict(self._totals)


class FakeAgent:
    """最小可用的 OrchestratorAgent 替身"""

    def __init__(self, draft=None, tool_log=None, report_md="# 报告", stop_reason="tool_use",
                 raises=None):
        self.last_draft = draft
        self.last_tool_log = tool_log or []
        self.last_compliance = {"draft": draft, "score": 80, "issues": [], "is_compliant": True}
        self.loop = types.SimpleNamespace(last_stop_reason=stop_reason)
        self.usage = FakeUsage({"calls": 3, "input_tokens": 100, "output_tokens": 50})
        self._report_md = report_md
        self._raises = raises

    def generate_report(self, entity, feedback_store=None):
        if self._raises:
            raise self._raises
        return f"sess-{entity}", self._report_md


def make_builder(agent_or_factory):
    """返回一个符合 build_agent 签名的假构造函数，并记录调用参数"""
    calls = []

    def build(variant, store, company, graph=None, temperature=None, **kwargs):
        calls.append({"variant": variant, "company": company, "temperature": temperature})
        if callable(agent_or_factory):
            return agent_or_factory(variant, company)
        return agent_or_factory

    build.calls = calls
    return build


DRAFT = {"entity": "AAPL", "summary": "摘要", "confidence": 0.6}
TOOL_LOG = [
    {"tool": "query_company_graph", "input": {"company": "AAPL"},
     "result": {"neighbors": [{"company": "TSM"}], "_graph": "x" * 5000, "_mermaid": "y" * 5000}},
    {"tool": "get_market_data", "input": {"ticker": "AAPL"}, "result": {"price": 1.0}},
]


# ── slim_tool_log ────────────────────────────────────────────────

def test_slim_tool_log_drops_internal_fields():
    slim = runner.slim_tool_log(TOOL_LOG)
    dumped = json.dumps(slim, ensure_ascii=False)
    assert '"_graph"' not in dumped and '"_mermaid"' not in dumped
    assert slim[0]["result"]["neighbors"] == [{"company": "TSM"}]
    assert slim[1]["result"] == {"price": 1.0}
    # 原始对象不被修改
    assert "_graph" in TOOL_LOG[0]["result"]


def test_slim_tool_log_tolerates_missing_result():
    assert runner.slim_tool_log([{"tool": "x"}])[0]["result"] == {}


# ── run_one ─────────────────────────────────────────────────────

def test_run_one_writes_result_file(tmp_path):
    build = make_builder(FakeAgent(draft=DRAFT, tool_log=TOOL_LOG))
    store = types.SimpleNamespace(data={"recorded_at": "2026-09-19T00:00:00Z"})

    record = runner.run_one("C", "AAPL", 1, store, graph=None, out_dir=tmp_path, build_agent=build)

    assert record["status"] == "ok"
    assert record["attempt"] == 1
    assert record["session_id"] == "sess-AAPL"
    assert record["draft"] == DRAFT
    assert record["usage"]["calls"] == 3
    assert record["stop_reason"] == "tool_use"
    assert record["snapshot_recorded_at"] == "2026-09-19T00:00:00Z"
    assert "draft" not in record["compliance"]          # 合规结果里的草稿不重复落盘
    assert record["compliance"]["score"] == 80

    path = runner.run_path(tmp_path, "C", "AAPL", 1)
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["company"] == "AAPL" and on_disk["variant"] == "C"
    assert '"_graph"' not in path.read_text(encoding="utf-8")
    # temperature 必须固定为 0，保证可复现
    assert build.calls[0]["temperature"] == 0


def test_run_one_marks_no_draft(tmp_path):
    build = make_builder(FakeAgent(draft=None, report_md="Agent 未能生成报告草稿"))
    store = types.SimpleNamespace(data={})
    record = runner.run_one("A", "MSFT", 1, store, None, tmp_path, build)
    assert record["status"] == "no_draft"
    assert runner.run_path(tmp_path, "A", "MSFT", 1).exists()


def test_run_one_retries_then_records_failure(tmp_path):
    attempts = []

    def factory(variant, company):
        attempts.append(company)
        return FakeAgent(raises=RuntimeError("API 超时"))

    store = types.SimpleNamespace(data={})
    record = runner.run_one("B", "NVDA", 2, store, None, tmp_path,
                            make_builder(factory), retries=1)

    assert len(attempts) == 2                     # 重试了一次
    assert record["status"] == "failed"
    assert record["attempt"] == 2
    assert "RuntimeError: API 超时" == record["error"]
    assert "traceback" in record
    assert runner.run_path(tmp_path, "B", "NVDA", 2).exists()


def test_run_one_succeeds_on_retry(tmp_path):
    state = {"n": 0}

    def factory(variant, company):
        state["n"] += 1
        if state["n"] == 1:
            return FakeAgent(raises=RuntimeError("第一次失败"))
        return FakeAgent(draft=DRAFT)

    record = runner.run_one("C", "AAPL", 1, types.SimpleNamespace(data={}), None,
                            tmp_path, make_builder(factory))
    assert record["status"] == "ok" and record["attempt"] == 2
    assert "error" in record                      # 保留首次失败的痕迹
    assert record["draft"] == DRAFT


# ── plan_runs（断点续跑）────────────────────────────────────────

def test_plan_runs_enumerates_all_combinations(tmp_path):
    jobs = runner.plan_runs(["A", "C"], ["AAPL", "MSFT"], 2, tmp_path, overwrite=False)
    assert len(jobs) == 2 * 2 * 2
    assert ("A", "AAPL", 1) in jobs and ("C", "MSFT", 2) in jobs


def test_plan_runs_skips_existing(tmp_path):
    done = runner.run_path(tmp_path, "A", "AAPL", 1)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text("{}", encoding="utf-8")

    jobs = runner.plan_runs(["A"], ["AAPL", "MSFT"], 1, tmp_path, overwrite=False)
    assert jobs == [("A", "MSFT", 1)]

    jobs = runner.plan_runs(["A"], ["AAPL", "MSFT"], 1, tmp_path, overwrite=True)
    assert len(jobs) == 2


# ── run_batch ───────────────────────────────────────────────────

@pytest.mark.parametrize("workers", [1, 2])
def test_run_batch_runs_every_job(tmp_path, workers):
    build = make_builder(lambda v, c: FakeAgent(draft={**DRAFT, "entity": c}))
    jobs = [("C", "AAPL", 1), ("C", "MSFT", 1), ("A", "AAPL", 1)]
    lines = []

    records = runner.run_batch(jobs, types.SimpleNamespace(data={}), None, tmp_path,
                               workers=workers, build_agent=build, on_progress=lines.append)

    assert len(records) == 3
    assert all(r["status"] == "ok" for r in records)
    assert len(lines) == 3
    assert all(runner.run_path(tmp_path, *j).exists() for j in jobs)


def test_run_batch_continues_after_failure(tmp_path):
    def factory(variant, company):
        if company == "MSFT":
            return FakeAgent(raises=ValueError("坏数据"))
        return FakeAgent(draft=DRAFT)

    jobs = [("C", "AAPL", 1), ("C", "MSFT", 1)]
    records = runner.run_batch(jobs, types.SimpleNamespace(data={}), None, tmp_path,
                               build_agent=make_builder(factory), on_progress=lambda s: None)
    assert [r["status"] for r in records] == ["ok", "failed"]


# ── 统计 ────────────────────────────────────────────────────────

def test_load_runs_and_summarize(tmp_path):
    build = make_builder(lambda v, c: FakeAgent(draft=DRAFT))
    jobs = [("C", "AAPL", 1), ("C", "MSFT", 1), ("A", "AAPL", 1)]
    runner.run_batch(jobs, types.SimpleNamespace(data={}), None, tmp_path,
                     build_agent=build, on_progress=lambda s: None)

    runs = runner.load_runs(tmp_path)
    assert len(runs) == 3

    summary = runner.summarize(runs)
    assert summary["C"]["runs"] == 2 and summary["C"]["ok"] == 2
    assert summary["C"]["companies"] == 2
    assert summary["C"]["avg_calls"] == 3.0
    assert summary["C"]["avg_tokens"] == 150
    assert summary["A"]["runs"] == 1

    lines = []
    runner.print_summary(summary, on_progress=lines.append)
    assert any("C" in line for line in lines)


def test_summarize_counts_failures():
    runs = [
        {"variant": "C", "company": "AAPL", "status": "ok", "wall_seconds": 10,
         "usage": {"calls": 2, "input_tokens": 10, "output_tokens": 5}},
        {"variant": "C", "company": "MSFT", "status": "failed", "wall_seconds": 1},
        {"variant": "C", "company": "NVDA", "status": "no_draft", "wall_seconds": 2},
    ]
    agg = runner.summarize(runs)["C"]
    assert (agg["ok"], agg["failed"], agg["no_draft"]) == (1, 1, 1)
    assert agg["runs"] == 3 and agg["companies"] == 3


# ── 命令行 ──────────────────────────────────────────────────────

def test_main_dry_run_lists_jobs_without_running(tmp_path, capsys, monkeypatch):
    companies = tmp_path / "companies.txt"
    companies.write_text("AAPL\nMSFT\n", encoding="utf-8")
    monkeypatch.setattr(runner.SnapshotStore, "load",
                        staticmethod(lambda *a, **k: pytest.fail("dry-run 不应加载快照")))

    runner.main(["--variant", "C", "--repeat", "2", "--companies", str(companies),
                 "--out", str(tmp_path / "runs"), "--dry-run"])

    out = capsys.readouterr().out
    assert "待执行 4 个任务" in out
    assert "C AAPL r1" in out
    assert not (tmp_path / "runs").exists()


def test_main_smoke_limits_scope(tmp_path, capsys):
    companies = tmp_path / "companies.txt"
    companies.write_text("AAPL\nMSFT\nNVDA\n", encoding="utf-8")
    runner.main(["--variant", "all", "--companies", str(companies),
                 "--out", str(tmp_path / "runs"), "--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert f"待执行 {2 * len(runner.variants.VARIANTS)} 个任务" in out


def test_main_only_filters_companies(tmp_path, capsys):
    companies = tmp_path / "companies.txt"
    companies.write_text("AAPL\nMSFT\nNVDA\n", encoding="utf-8")
    runner.main(["--variant", "C", "--repeat", "1", "--companies", str(companies),
                 "--only", "nvda", "--out", str(tmp_path / "runs"), "--dry-run"])
    out = capsys.readouterr().out
    assert "待执行 1 个任务" in out and "C NVDA r1" in out


def test_main_summary_reports_existing_runs(tmp_path, capsys):
    build = make_builder(lambda v, c: FakeAgent(draft=DRAFT))
    out_dir = tmp_path / "runs"
    runner.run_batch([("C", "AAPL", 1)], types.SimpleNamespace(data={}), None, out_dir,
                     build_agent=build, on_progress=lambda s: None)
    runner.main(["--summary", "--out", str(out_dir)])
    assert "C" in capsys.readouterr().out


def test_main_summary_without_results(tmp_path, capsys):
    runner.main(["--summary", "--out", str(tmp_path / "empty")])
    assert "没有结果文件" in capsys.readouterr().out


# ── load_runs 的健壮性（报告目录与结果目录同级）──────────────────

def test_load_runs_ignores_report_and_graph_files(tmp_path):
    build = make_builder(lambda v, c: FakeAgent(draft=DRAFT))
    runner.run_batch([("C", "AAPL", 1)], types.SimpleNamespace(data={}), None, tmp_path,
                     build_agent=build, on_progress=lambda s: None)

    # Orchestrator 会把报告与图谱写进 out_dir/_reports/
    reports = tmp_path / "_reports"
    reports.mkdir()
    (reports / "AAPL_20260919_180658_graph.json").write_text(
        json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    (reports / "AAPL_20260919_180658.md").write_text("# 报告", encoding="utf-8")

    runs = runner.load_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["variant"] == "C"


def test_load_runs_skips_unparsable_and_foreign_json(tmp_path, capsys):
    build = make_builder(lambda v, c: FakeAgent(draft=DRAFT))
    runner.run_batch([("C", "AAPL", 1)], types.SimpleNamespace(data={}), None, tmp_path,
                     build_agent=build, on_progress=lambda s: None)
    (tmp_path / "C" / "notes.json").write_text("{坏的", encoding="utf-8")
    (tmp_path / "C" / "other.json").write_text('{"foo": 1}', encoding="utf-8")

    runs = runner.load_runs(tmp_path)
    assert len(runs) == 1
    assert "跳过无法解析" in capsys.readouterr().out


def test_summary_cli_survives_report_dir(tmp_path, capsys):
    """回归：eval/runs/_reports/*_graph.json 曾导致 --summary KeyError"""
    build = make_builder(lambda v, c: FakeAgent(draft=DRAFT))
    out_dir = tmp_path / "runs"
    runner.run_batch([("C", "AAPL", 1)], types.SimpleNamespace(data={}), None, out_dir,
                     build_agent=build, on_progress=lambda s: None)
    (out_dir / "_reports").mkdir()
    (out_dir / "_reports" / "AAPL_graph.json").write_text('{"nodes": []}', encoding="utf-8")

    runner.main(["--summary", "--out", str(out_dir)])
    assert "C" in capsys.readouterr().out
