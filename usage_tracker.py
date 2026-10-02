"""
usage_tracker.py
-----------------
统计一次报告生成过程中所有 Anthropic API 调用的次数、token 与耗时。

做法：用 TrackedClient 包住 anthropic client，而不是在每个调用点各写一遍统计。
OrchestratorLoop / CriticAgent / EventExtractor 共用同一个 client，
包一层就能覆盖全部调用（包括并行执行的事件抽取）。

用途：消融实验要比较各方案的成本（见 docs/eval-design.md）；
日常运行时统计结果会写入反馈存储的快照，便于事后回看单份报告的开销。
"""

import threading
import time
from typing import Optional


class UsageTracker:
    """线程安全的调用统计（事件抽取是多线程并行的）"""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    def record(self, model: Optional[str], usage, seconds: float, error: Optional[str] = None):
        entry = {
            "model": model,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "seconds": round(seconds, 3),
        }
        if error:
            entry["error"] = error
        with self._lock:
            self.calls.append(entry)

    def totals(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        return {
            "calls":         len(calls),
            "failed_calls":  sum(1 for c in calls if "error" in c),
            "input_tokens":  sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "seconds":       round(sum(c["seconds"] for c in calls), 3),
        }

    def by_model(self) -> dict:
        result: dict[str, dict] = {}
        with self._lock:
            calls = list(self.calls)
        for c in calls:
            agg = result.setdefault(c["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            agg["calls"] += 1
            agg["input_tokens"] += c["input_tokens"]
            agg["output_tokens"] += c["output_tokens"]
        return result

    def reset(self):
        with self._lock:
            self.calls.clear()


class _TrackedMessages:
    def __init__(self, client, tracker: UsageTracker):
        self._client = client
        self._tracker = tracker

    def create(self, **kwargs):
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as e:
            self._tracker.record(kwargs.get("model"), None, time.perf_counter() - started, error=str(e))
            raise
        self._tracker.record(kwargs.get("model"), getattr(response, "usage", None),
                             time.perf_counter() - started)
        return response

    def __getattr__(self, name):        # 透传 create 之外的方法（如 stream）
        return getattr(self._client.messages, name)


class TrackedClient:
    """
    anthropic client 的透明代理：调用方式不变，只是顺带记录 usage。

    >>> tracker = UsageTracker()
    >>> client = TrackedClient(anthropic.Anthropic(api_key="..."), tracker)
    >>> client.messages.create(...)
    >>> tracker.totals()
    {'calls': 1, 'failed_calls': 0, 'input_tokens': 1200, 'output_tokens': 300, 'seconds': 1.2}
    """

    def __init__(self, client, tracker: Optional[UsageTracker] = None):
        self._client = client
        self.usage = tracker or UsageTracker()
        self._messages = _TrackedMessages(client, self.usage)

    @property
    def messages(self):
        return self._messages

    def __getattr__(self, name):
        return getattr(self._client, name)
