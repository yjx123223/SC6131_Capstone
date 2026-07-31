"""
tests/test_feedback_tools.py
------------------------------
tools.feedback_tools.get_feedback_stats 的行为测试。

用真实的 FeedbackStore（指向 tmp_path 下的临时 sqlite 文件）验证，
因为 FeedbackStore 本身逻辑很轻，没必要再 stub。
"""

from feedback_store import FeedbackStore
from tools import feedback_tools, resources


def _make_store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "feedback.db")


def test_get_feedback_stats_no_ratings_yet(tmp_path):
    store = _make_store(tmp_path)
    result = feedback_tools.get_feedback_stats(store=store)

    assert result["total_rated"] == 0
    assert result["positive_rate"] == 0.0
    assert result["signal_stats"] == {}


def test_get_feedback_stats_aggregates_ratings(tmp_path):
    store = _make_store(tmp_path)

    summary_positive = {
        "period": "2022-01-03 ~ 2022-01-17",
        "total_events": 2,
        "positive_impacts": [{"subject": "Goldman Sachs Group", "relation": "Positive_Impact_On", "object": "Apple Inc.", "count": 1}],
        "negative_impacts": [],
    }
    summary_negative = {
        "period": "2022-01-10 ~ 2022-01-24",
        "total_events": 1,
        "positive_impacts": [],
        "negative_impacts": [{"subject": "Meta Platforms", "relation": "Negative_Impact_On", "object": "Apple Inc.", "count": 1}],
    }

    sid1 = store.log_advice("Apple Inc.", summary_positive, "建议1")
    sid2 = store.log_advice("Apple Inc.", summary_negative, "建议2")
    store.rate(sid1, rating=1)
    store.rate(sid2, rating=-1)

    result = feedback_tools.get_feedback_stats(store=store)

    assert result["total_rated"] == 2
    assert result["positive_rate"] == 0.5
    assert result["signal_stats"]["Positive_Impact_On"]["avg_rating"] == 1.0
    assert result["signal_stats"]["Negative_Impact_On"]["avg_rating"] == -1.0


def test_get_feedback_stats_filters_by_relation_type(tmp_path):
    store = _make_store(tmp_path)
    summary = {
        "period": "p",
        "total_events": 1,
        "positive_impacts": [{"subject": "S", "relation": "Positive_Impact_On", "object": "O", "count": 1}],
        "negative_impacts": [{"subject": "S", "relation": "Negative_Impact_On", "object": "O", "count": 1}],
    }
    sid = store.log_advice("Apple Inc.", summary, "建议")
    store.rate(sid, rating=1)

    result = feedback_tools.get_feedback_stats(relation_type="positive", store=store)

    assert "Positive_Impact_On" in result["signal_stats"]
    assert "Negative_Impact_On" not in result["signal_stats"]


def test_get_feedback_stats_error_is_reported_not_raised(tmp_path):
    store = _make_store(tmp_path)

    class BoomStore:
        def signal_accuracy_report(self):
            raise RuntimeError("db is locked")

    result = feedback_tools.get_feedback_stats(store=BoomStore())
    assert "error" in result


def test_default_store_falls_back_to_resources_singleton(tmp_path, monkeypatch):
    """未显式传入 store 时，应该走 tools.resources.get_store() 的单例"""
    resources.reset()
    fake_store = _make_store(tmp_path)
    monkeypatch.setattr(resources, "get_store", lambda: fake_store)

    result = feedback_tools.get_feedback_stats()
    assert result["total_rated"] == 0
