"""
tests/test_main.py
-------------------
命令行入口的参数分发与评分交互（替换掉真实的 Orchestrator，不调用 API）。
"""

import pytest

import main
import orchestrator


class FakeOrch:
    calls = []

    def generate_report(self, entity, feedback_store=None):
        FakeOrch.calls.append(("report", entity))
        return 7, f"# 报告 {entity}"

    def generate_comparison_report(self, entities, feedback_store=None):
        FakeOrch.calls.append(("compare", tuple(entities)))
        return "# 对比"


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    import config
    FakeOrch.calls = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", tmp_path / "f.db")
    monkeypatch.setattr(orchestrator, "OrchestratorAgent", FakeOrch)


def test_entity_runs_report_and_rates(monkeypatch, capsys):
    rated = []
    import feedback_store
    monkeypatch.setattr(feedback_store.FeedbackStore, "rate",
                        lambda self, sid, r, note="": rated.append((sid, r, note)))
    answers = iter(["+1", "不错"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    main.main(["--entity", "NVDA"])

    assert FakeOrch.calls == [("report", "NVDA")]
    assert "# 报告 NVDA" in capsys.readouterr().out
    assert rated == [(7, 1, "不错")]


@pytest.mark.parametrize("answer, expected", [("", []), ("0", [(7, 0, "")])])
def test_rating_skip_and_neutral(monkeypatch, answer, expected):
    rated = []
    import feedback_store
    monkeypatch.setattr(feedback_store.FeedbackStore, "rate",
                        lambda self, sid, r, note="": rated.append((sid, r, note)))
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    main.main(["--entity", "AAPL"])
    assert rated == expected


def test_legacy_multi_agent_flag_still_works(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    main.main(["--multi-agent", "--entity", "AAPL"])
    assert FakeOrch.calls == [("report", "AAPL")]


def test_compare(capsys):
    main.main(["--compare", "AAPL", "MSFT"])
    assert FakeOrch.calls == [("compare", ("AAPL", "MSFT"))]
    assert "# 对比" in capsys.readouterr().out


def test_history_does_not_need_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    main.main(["--history", "--entity", "AAPL"])
    assert "暂无历史记录" in capsys.readouterr().out
    assert FakeOrch.calls == []


def test_missing_api_key_exits(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(SystemExit):
        main.main(["--entity", "AAPL"])


def test_no_args_prints_help(capsys):
    main.main([])
    assert "--entity" in capsys.readouterr().out
