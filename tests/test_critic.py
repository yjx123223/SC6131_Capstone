"""
tests/test_critic.py
-----------------------
critic.CriticAgent.review 的行为测试（拆分自 orchestrator.py 的
OrchestratorAgent._run_critic）。用 stub 的 anthropic client 驱动，
不发真实网络请求。
"""

from critic import CriticAgent


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, reply_text):
        self._reply_text = reply_text
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeResponse(self._reply_text)


class _FakeClient:
    def __init__(self, reply_text):
        self.messages = _FakeMessages(reply_text)


_DRAFT = {
    "executive_summary": "summary",
    "macro_analysis": "macro",
    "entity_analysis": "entity",
    "recommendation": "增持",
    "recommendation_rationale": "理由",
    "risk_warnings": "风险",
    "confidence": "high",
    "key_signals": ["signal1"],
}


def test_review_parses_valid_json_response():
    reply = '{"approved": false, "conflicts": ["置信度过高"], "confidence_adjustment": "lower", "suggestions": "降低置信度"}'
    client = _FakeClient(reply)
    critic = CriticAgent(client, model="claude-haiku-4-5", max_tokens=1024)

    result = critic.review("Apple Inc.", _DRAFT, tool_log=[])

    assert result["approved"] is False
    assert result["conflicts"] == ["置信度过高"]
    assert result["confidence_adjustment"] == "lower"


def test_review_extracts_json_even_with_surrounding_text():
    reply = '这是我的审查结果：\n{"approved": true, "conflicts": [], "confidence_adjustment": "maintain", "suggestions": ""}\n谢谢'
    client = _FakeClient(reply)
    critic = CriticAgent(client, model="claude-haiku-4-5", max_tokens=1024)

    result = critic.review("Apple Inc.", _DRAFT, tool_log=[])
    assert result["approved"] is True


def test_review_falls_back_to_approved_when_response_is_not_json():
    """
    已知行为：Claude 返回非 JSON 内容时，当前策略是默认放行
    （approved=True）。这段测试把这个行为显式钉住，以后如果决定收紧
    （比如改成 approved=False + lower），这个测试会先失败提醒改动者。
    """
    client = _FakeClient("我拒绝以 JSON 格式回复。")
    critic = CriticAgent(client, model="claude-haiku-4-5", max_tokens=1024)

    result = critic.review("Apple Inc.", _DRAFT, tool_log=[])

    assert result["approved"] is True
    assert result["conflicts"] == []


def test_review_uses_configured_model_and_max_tokens():
    client = _FakeClient('{"approved": true, "conflicts": [], "confidence_adjustment": "maintain", "suggestions": ""}')
    critic = CriticAgent(client, model="custom-model", max_tokens=999)

    critic.review("Apple Inc.", _DRAFT, tool_log=[])

    assert client.messages.last_call_kwargs["model"] == "custom-model"
    assert client.messages.last_call_kwargs["max_tokens"] == 999


def test_summarize_tool_log_includes_kg_and_macro_signals():
    tool_log = [
        {
            "tool": "query_kg_signals",
            "input": {},
            "result": {"entity": "Apple Inc.", "total_events": 4, "positive_impacts": [1], "negative_impacts": []},
        },
        {
            "tool": "query_macro",
            "input": {},
            "result": {"summary_text": "VIX: 13.2"},
        },
    ]
    summary = CriticAgent._summarize_tool_log(tool_log)

    assert "Apple Inc." in summary
    assert "VIX: 13.2" in summary
