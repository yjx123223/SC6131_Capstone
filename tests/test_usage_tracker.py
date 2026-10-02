"""
tests/test_usage_tracker.py
----------------------------
UsageTracker / TrackedClient：调用统计与透明代理行为。
"""

import threading

import pytest

from usage_tracker import TrackedClient, UsageTracker


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Resp:
    def __init__(self, i=100, o=20):
        self.usage = _Usage(i, o)
        self.content = []


class _Messages:
    def __init__(self, exc=None):
        self.calls = []
        self._exc = exc

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return _Resp()

    def stream(self, **kwargs):
        return "streamed"


class _Client:
    def __init__(self, exc=None):
        self.messages = _Messages(exc)
        self.api_key = "k"


def test_records_calls_and_totals():
    client = TrackedClient(_Client())
    client.messages.create(model="claude-haiku-4-5", messages=[])
    client.messages.create(model="claude-haiku-4-5", messages=[])

    totals = client.usage.totals()
    assert totals["calls"] == 2
    assert totals["input_tokens"] == 200 and totals["output_tokens"] == 40
    assert totals["failed_calls"] == 0
    assert totals["seconds"] >= 0


def test_passes_arguments_through_and_returns_response():
    inner = _Client()
    client = TrackedClient(inner)
    resp = client.messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "x"}])

    assert resp.usage.input_tokens == 100
    assert inner.messages.calls[0]["max_tokens"] == 8
    assert client.messages.stream() == "streamed"    # 其他方法透传
    assert client.api_key == "k"                     # 其他属性透传


def test_failed_call_is_recorded_and_reraised():
    client = TrackedClient(_Client(exc=RuntimeError("overloaded")))
    with pytest.raises(RuntimeError):
        client.messages.create(model="m", messages=[])

    totals = client.usage.totals()
    assert totals["calls"] == 1 and totals["failed_calls"] == 1
    assert client.usage.calls[0]["error"] == "overloaded"
    assert totals["input_tokens"] == 0


def test_response_without_usage_counts_zero_tokens():
    class NoUsage:
        pass

    class M:
        def create(self, **kw):
            return NoUsage()

    class C:
        messages = M()

    client = TrackedClient(C())
    client.messages.create(model="m")
    assert client.usage.totals() == {"calls": 1, "failed_calls": 0, "input_tokens": 0,
                                     "output_tokens": 0, "seconds": client.usage.calls[0]["seconds"]}


def test_by_model_aggregation():
    client = TrackedClient(_Client())
    client.messages.create(model="a", messages=[])
    client.messages.create(model="b", messages=[])
    client.messages.create(model="a", messages=[])

    assert client.usage.by_model() == {
        "a": {"calls": 2, "input_tokens": 200, "output_tokens": 40},
        "b": {"calls": 1, "input_tokens": 100, "output_tokens": 20},
    }


def test_thread_safe_under_parallel_calls():
    client = TrackedClient(_Client())

    def work():
        for _ in range(20):
            client.messages.create(model="m", messages=[])

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert client.usage.totals()["calls"] == 120


def test_reset():
    client = TrackedClient(_Client())
    client.messages.create(model="m")
    client.usage.reset()
    assert client.usage.totals()["calls"] == 0
