"""
tests/test_live_kg_extractor.py
--------------------------------
live_kg.event_extractor.EventExtractor：用假的 anthropic client 驱动，
覆盖请求构造、结构化结果校验、实体链接与各类失败分支。
"""

import pytest

from live_kg.event_extractor import EventExtractor, EXTRACT_TOOL

ARTICLES = [
    {"title": "TSMC warns of capacity constraints", "publisher": "Bloomberg",
     "published_at": "2026-09-12T08:00+00:00", "summary": "Advanced node capacity tight.", "url": "u0"},
    {"title": "Apple and TSMC extend chip deal", "publisher": "Reuters",
     "published_at": "2026-09-10T08:00+00:00", "summary": "", "url": "u1"},
]
KNOWN = {"TSM": "Taiwan Semiconductor", "AAPL": "Apple Inc.", "NVDA": "NVIDIA"}


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason


class _Client:
    def __init__(self, response=None, exc=None):
        self._response, self._exc = response, exc
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return self._response


def _tool_resp(events):
    return _Resp([_Block(type="tool_use", name="record_events", input={"events": events})])


def _ev(**overrides):
    base = {"article_index": 0, "event_type": "SUPPLY_CHAIN", "polarity": "negative",
            "affected_companies": ["TSM"], "summary": "台积电先进制程产能吃紧", "confidence": 0.8}
    base.update(overrides)
    return base


def test_request_forces_tool_and_lists_known_companies():
    client = _Client(_tool_resp([]))
    EventExtractor(client, model="m", max_tokens=123).extract("tsm", ARTICLES, KNOWN)

    call = client.calls[0]
    assert call["model"] == "m" and call["max_tokens"] == 123
    assert call["tool_choice"] == {"type": "tool", "name": "record_events"}
    assert call["tools"] == [EXTRACT_TOOL]
    prompt = call["messages"][0]["content"]
    assert "新闻所属公司：TSM（Taiwan Semiconductor）" in prompt
    assert "AAPL（Apple Inc.）" in prompt
    assert "[0] 2026-09-12 | Bloomberg | TSMC warns of capacity constraints" in prompt
    assert "    Advanced node capacity tight." in prompt
    assert "SUPPLY_CHAIN（供应链）" in prompt


def test_valid_event_is_normalized():
    client = _Client(_tool_resp([_ev()]))
    result = EventExtractor(client).extract("TSM", ARTICLES, KNOWN)

    assert result["dropped"] == {}
    assert "error" not in result
    assert result["events"] == [{
        "article_index": 0, "event_type": "SUPPLY_CHAIN", "polarity": "negative",
        "tickers": ["TSM"], "summary": "台积电先进制程产能吃紧", "confidence": 0.8,
        "date": "2026-09-12",
    }]


def test_company_names_are_linked_and_deduped():
    raw = _ev(article_index=1, polarity="positive",
              affected_companies=["Apple", "Taiwan Semiconductor Manufacturing Company", "tsm", "Samsung"])
    result = EventExtractor(_Client(_tool_resp([raw]))).extract("TSM", ARTICLES, KNOWN)

    assert result["events"][0]["tickers"] == ["AAPL", "TSM"]
    assert result["events"][0]["date"] == "2026-09-10"
    assert result["dropped"] == {"公司无法链接": 1}


def test_company_resolvable_but_not_in_graph_is_dropped():
    raw = _ev(affected_companies=["Microsoft"])       # 能解析成 MSFT，但不在已知公司里
    result = EventExtractor(_Client(_tool_resp([raw]))).extract("TSM", ARTICLES, KNOWN)
    assert result["events"] == []
    assert result["dropped"] == {"公司无法链接": 1, "无可链接的公司": 1}


def test_missing_companies_default_to_focus_company():
    raw = _ev(affected_companies=[])
    result = EventExtractor(_Client(_tool_resp([raw]))).extract("TSM", ARTICLES, KNOWN)
    assert result["events"][0]["tickers"] == ["TSM"]


@pytest.mark.parametrize("overrides, reason", [
    ({"article_index": 5}, "新闻编号无效"),
    ({"article_index": -1}, "新闻编号无效"),
    ({"article_index": "0"}, "新闻编号无效"),
    ({"article_index": True}, "新闻编号无效"),
    ({"polarity": "bullish"}, "正负面无效"),
])
def test_invalid_events_are_dropped(overrides, reason):
    result = EventExtractor(_Client(_tool_resp([_ev(**overrides)]))).extract("TSM", ARTICLES, KNOWN)
    assert result["events"] == []
    assert result["dropped"] == {reason: 1}


def test_unknown_event_type_becomes_other():
    result = EventExtractor(_Client(_tool_resp([_ev(event_type="RUMOR")]))).extract("TSM", ARTICLES, KNOWN)
    assert result["events"][0]["event_type"] == "OTHER"
    assert result["dropped"] == {"事件类型归为 OTHER": 1}


def test_confidence_and_summary_are_sanitized():
    raw = _ev(confidence="high", summary="")
    raw2 = _ev(confidence=7, summary="x" * 200)
    result = EventExtractor(_Client(_tool_resp([raw, raw2, "garbage"]))).extract("TSM", ARTICLES, KNOWN)

    e1, e2 = result["events"]
    assert e1["confidence"] == 0.5
    assert e1["summary"] == "TSMC warns of capacity constraints"   # 摘要为空时用新闻标题
    assert e2["confidence"] == 1.0
    assert len(e2["summary"]) == 80
    assert result["dropped"] == {"格式错误": 1}


def test_default_known_companies_is_focus_only():
    raw = _ev(affected_companies=["TSM", "AAPL"])
    result = EventExtractor(_Client(_tool_resp([raw]))).extract("TSM", ARTICLES)
    assert result["events"][0]["tickers"] == ["TSM"]


def test_no_articles_skips_api_call():
    client = _Client(_tool_resp([]))
    result = EventExtractor(client).extract("TSM", [], KNOWN)
    assert result == {"events": [], "dropped": {}}
    assert client.calls == []


def test_api_exception_returns_error():
    result = EventExtractor(_Client(exc=RuntimeError("overloaded"))).extract("TSM", ARTICLES, KNOWN)
    assert result["events"] == [] and "overloaded" in result["error"]


def test_truncated_output_returns_error():
    resp = _Resp([_Block(type="tool_use", name="record_events", input={"events": [_ev()]})], stop_reason="max_tokens")
    result = EventExtractor(_Client(resp)).extract("TSM", ARTICLES, KNOWN)
    assert result["events"] == [] and "截断" in result["error"]


@pytest.mark.parametrize("content", [
    [_Block(type="text", text="no tool")],
    [_Block(type="tool_use", name="record_events", input={"events": "oops"})],
])
def test_missing_structured_output_returns_error(content):
    result = EventExtractor(_Client(_Resp(content))).extract("TSM", ARTICLES, KNOWN)
    assert result["events"] == [] and "未返回结构化结果" in result["error"]
