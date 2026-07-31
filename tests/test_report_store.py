"""
tests/test_report_store.py
------------------------------
report_store.save_report 的行为测试（拆分自
orchestrator.py 的 OrchestratorAgent._save_report）。
"""

from report_store import save_report


def test_save_report_writes_file_with_sanitized_name(tmp_path):
    path = save_report("Apple Inc.", "# 报告内容", reports_dir=tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.read_text(encoding="utf-8") == "# 报告内容"
    # 实体名中的空格/句点应该被替换成下划线，文件名保持安全
    assert "Apple_Inc" in path.name


def test_save_report_creates_reports_dir_if_missing(tmp_path):
    nested_dir = tmp_path / "nested" / "reports"
    path = save_report("Microsoft Corporation", "内容", reports_dir=nested_dir)

    assert nested_dir.exists()
    assert path.exists()


def test_save_report_uses_config_reports_dir_by_default(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)

    path = save_report("Apple Inc.", "内容")
    assert path.parent == tmp_path
