"""
tests/test_feedback_store.py
------------------------------
FeedbackStore 的写入、评分与查询。
"""

import sqlite3

import pytest

import config
from feedback_store import FeedbackStore


def _make_store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "feedback.db")


def test_log_advice_and_get_history(tmp_path):
    store = _make_store(tmp_path)
    snapshot = {"period": "行情截至 2026-09-15", "total_events": 5, "recommendation": "持有"}

    sid = store.log_advice("Apple Inc.", snapshot, "报告文本", model="claude-haiku-4-5")
    assert isinstance(sid, int)

    history = store.get_history("Apple Inc.")
    assert len(history) == 1
    rec = history[0]
    assert rec["entity"] == "Apple Inc."
    assert rec["advice_text"] == "报告文本"
    assert rec["period"] == "行情截至 2026-09-15"
    assert rec["total_events"] == 5
    assert rec["snapshot"]["recommendation"] == "持有"
    assert rec["rating"] is None
    assert "kg_summary" not in rec and "time_window" not in rec


def test_rate_updates_row(tmp_path):
    store = _make_store(tmp_path)
    sid = store.log_advice("Apple Inc.", {}, "报告")
    store.rate(sid, rating=1, note="判断准确")

    history = store.get_history("Apple Inc.")
    assert history[0]["rating"] == 1
    assert history[0]["note"] == "判断准确"


def test_rate_rejects_invalid_value(tmp_path):
    store = _make_store(tmp_path)
    sid = store.log_advice("Apple Inc.", {}, "报告")
    with pytest.raises(ValueError):
        store.rate(sid, rating=2)


def test_get_history_filters_by_entity_and_limit(tmp_path):
    store = _make_store(tmp_path)
    store.log_advice("Apple Inc.", {}, "A1")
    store.log_advice("Microsoft Corporation", {}, "B")
    store.log_advice("Apple Inc.", {}, "A2")

    assert [r["entity"] for r in store.get_history("Apple Inc.")] == ["Apple Inc.", "Apple Inc."]
    assert len(store.get_history(limit=2)) == 2


def test_reads_legacy_rows(tmp_path):
    """早期版本写入的记录（time_window 非 0、kg_summary 为 FinDKG 摘要）仍可正常读取"""
    store = _make_store(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "INSERT INTO advice_sessions (entity, time_window, period, total_events, kg_summary, advice_text, created_at) "
            "VALUES ('Apple Inc.', 12, '2022-10-23 ~ 2022-12-25', 86, '{\"positive_impacts\": []}', '旧报告', 0)"
        )
        conn.execute(
            "INSERT INTO advice_sessions (entity, advice_text, created_at) VALUES ('Apple Inc.', '空快照', 1)"
        )
    history = store.get_history("Apple Inc.")
    assert history[0]["snapshot"] == {}
    assert history[1]["snapshot"] == {"positive_impacts": []}


def test_print_history(tmp_path, capsys):
    store = _make_store(tmp_path)
    store.print_history()
    assert "暂无历史记录" in capsys.readouterr().out

    sid = store.log_advice("Apple Inc.", {"period": "行情截至 2026-09-15", "total_events": 3,
                                          "recommendation": "持有", "confidence": "medium"}, "报告")
    store.log_advice("Apple Inc.", {"period": "旧记录"}, "旧报告")
    store.rate(sid, 1, "准确")
    store.print_history("Apple Inc.")
    out = capsys.readouterr().out
    assert "建议：持有（置信度 medium）  行情截至 2026-09-15  新闻 3 条" in out
    assert "👍" in out and "备注：准确" in out
    assert "  旧记录" in out


def test_default_db_path_comes_from_config(tmp_path, monkeypatch):
    fake_path = tmp_path / "custom_feedback.db"
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", fake_path)

    store = FeedbackStore()
    assert store.db_path == fake_path
    assert fake_path.exists()
