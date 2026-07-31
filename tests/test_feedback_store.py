"""
tests/test_feedback_store.py
------------------------------
FeedbackStore 本身的 CRUD 基线测试（不经过 tools 层）。
"""

import pytest

import config
from feedback_store import FeedbackStore


def _make_store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "feedback.db")


def test_log_advice_and_get_history(tmp_path):
    store = _make_store(tmp_path)
    summary = {"period": "2022-01-03 ~ 2022-01-17", "total_events": 2}

    sid = store.log_advice("Apple Inc.", summary, "建议文本", model="claude-haiku-4-5")
    assert isinstance(sid, int)

    history = store.get_history("Apple Inc.")
    assert len(history) == 1
    assert history[0]["entity"] == "Apple Inc."
    assert history[0]["advice_text"] == "建议文本"
    assert history[0]["kg_summary"]["total_events"] == 2
    assert history[0]["rating"] is None


def test_rate_updates_row(tmp_path):
    store = _make_store(tmp_path)
    sid = store.log_advice("Apple Inc.", {}, "建议")
    store.rate(sid, rating=1, note="预测准确")

    history = store.get_history("Apple Inc.")
    assert history[0]["rating"] == 1
    assert history[0]["note"] == "预测准确"


def test_rate_rejects_invalid_value(tmp_path):
    store = _make_store(tmp_path)
    sid = store.log_advice("Apple Inc.", {}, "建议")
    with pytest.raises(ValueError):
        store.rate(sid, rating=2)


def test_signal_accuracy_report_empty(tmp_path):
    store = _make_store(tmp_path)
    report = store.signal_accuracy_report()
    assert report == {"total_rated": 0, "positive_rate": 0.0, "signal_stats": {}}


def test_get_history_filters_by_entity(tmp_path):
    store = _make_store(tmp_path)
    store.log_advice("Apple Inc.", {}, "建议A")
    store.log_advice("Microsoft Corporation", {}, "建议B")

    apple_only = store.get_history("Apple Inc.")
    assert len(apple_only) == 1
    assert apple_only[0]["entity"] == "Apple Inc."


def test_default_db_path_comes_from_config(tmp_path, monkeypatch):
    """
    不传 db_path 时，FeedbackStore 应该读取 config.FEEDBACK_DB_PATH，
    而不是自己维护一份路径（这是本次配置统一要修复的问题）。
    """
    fake_path = tmp_path / "custom_feedback.db"
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", fake_path)

    store = FeedbackStore()
    assert store.db_path == fake_path
    assert fake_path.exists()
