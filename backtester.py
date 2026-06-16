"""
backtester.py
-------------
基于 FinDKG 历史数据的 KG 信号回测。

核心逻辑：
  1. 选定历史时间点 T
  2. 用 T 之前的数据生成 KG 摘要（模拟当时的决策信号）
  3. 用 T 之后 N 周的数据验证：正面信号是否持续出现？负面信号是否兑现？
  4. 输出信号一致性得分（替代真实价格回报）

注意：这不是真实收益回测，是"KG 信号持续性"验证。
"""

from __future__ import annotations
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    entity: str
    decision_date: str          # T 时间点（周）
    n_recent_weeks: int         # 决策时看了多少周历史
    n_forward_weeks: int        # 验证未来多少周
    signal_at_T: dict           # T 时刻的 KG 摘要
    signal_direction: str       # "positive" / "negative" / "mixed" / "neutral"
    forward_positive: int       # T+N 周内出现的正面事件数
    forward_negative: int       # T+N 周内出现的负面事件数
    consistency_score: float    # [-1, 1]，正值表示预测方向与后续信号一致
    detail: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        direction_zh = {
            "positive": "正面",
            "negative": "负面",
            "mixed": "混合",
            "neutral": "中性",
        }.get(self.signal_direction, self.signal_direction)

        consistency_str = (
            "✅ 一致" if self.consistency_score > 0.2
            else "❌ 相反" if self.consistency_score < -0.2
            else "⚪ 无明显关联"
        )

        return (
            f"实体：{self.entity}\n"
            f"决策时间点：{self.decision_date}（看前{self.n_recent_weeks}周）\n"
            f"验证窗口：T+{self.n_forward_weeks}周\n"
            f"T 时信号方向：{direction_zh}\n"
            f"后续正面事件：{self.forward_positive}  负面事件：{self.forward_negative}\n"
            f"信号一致性：{consistency_str}（得分 {self.consistency_score:+.3f}）"
        )


class Backtester:
    """
    KG 信号历史回测器。

    使用示例
    --------
    >>> from kg_query import FinDKGGraph
    >>> from backtester import Backtester
    >>>
    >>> graph = FinDKGGraph()
    >>> bt = Backtester(graph)
    >>> result = bt.run("Apple Inc.", decision_date="2021-06-06")
    >>> print(result.summary())
    >>>
    >>> # 批量回测多个时间点
    >>> report = bt.rolling_backtest("Apple Inc.", n_windows=10)
    >>> print(report)
    """

    POSITIVE_RELS = {"Positive_Impact_On", "Raise", "Invests_In"}
    NEGATIVE_RELS = {"Negative_Impact_On", "Decrease"}

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : FinDKGGraph 实例
        """
        self.graph = graph

    # ── 核心回测 ─────────────────────────────────────────────────

    def run(
        self,
        entity: str,
        decision_date: str,
        n_recent_weeks: int = 12,
        n_forward_weeks: int = 8,
    ) -> BacktestResult:
        """
        单点回测。

        Parameters
        ----------
        entity          : 实体名
        decision_date   : 决策时间点，格式 "YYYY-MM-DD"（需在数据集时间范围内）
        n_recent_weeks  : 决策时向前看多少周
        n_forward_weeks : 向后验证多少周
        """
        g = self.graph

        # 找最近的 time_id
        t_id = self._find_closest_time_id(decision_date)
        if t_id is None:
            raise ValueError(f"找不到接近 {decision_date} 的时间节点，"
                             f"数据范围：{g.id2time.get(0)} ~ {g.id2time.get(g.max_time_id)}")

        actual_date = g.id2time.get(t_id, decision_date)
        eid = g.entity2id.get(entity)
        if eid is None:
            raise ValueError(f"找不到实体：{entity}")

        # 1. T 时刻的 KG 摘要（仅用 T 之前的数据）
        t_min = max(0, t_id - n_recent_weeks)
        signal_summary = self._compute_summary(eid, t_min, t_id)

        # 2. T+N 的 KG 事件（验证窗口）
        t_forward_max = min(g.max_time_id, t_id + n_forward_weeks)
        forward_df = g.triples[
            (g.triples["t"] > t_id) &
            (g.triples["t"] <= t_forward_max) &
            ((g.triples["s"] == eid) | (g.triples["o"] == eid))
        ]

        forward_positive = int(forward_df[
            forward_df["r"].map(g.id2relation).isin(self.POSITIVE_RELS)
        ].shape[0])
        forward_negative = int(forward_df[
            forward_df["r"].map(g.id2relation).isin(self.NEGATIVE_RELS)
        ].shape[0])

        # 3. 信号方向判断（基于 T 时刻）
        pos_at_T = signal_summary["pos_count"]
        neg_at_T = signal_summary["neg_count"]
        signal_direction = self._judge_direction(pos_at_T, neg_at_T)

        # 4. 一致性得分
        consistency_score = self._compute_consistency(
            signal_direction, forward_positive, forward_negative
        )

        # 5. 后续事件详情（供审查）
        detail_rows = []
        for _, row in forward_df.head(20).iterrows():
            detail_rows.append({
                "subject": g.id2entity.get(int(row["s"]), str(row["s"])),
                "relation": g.id2relation.get(int(row["r"]), str(row["r"])),
                "object": g.id2entity.get(int(row["o"]), str(row["o"])),
                "date": g.id2time.get(int(row["t"]), str(row["t"])),
            })

        return BacktestResult(
            entity=entity,
            decision_date=actual_date,
            n_recent_weeks=n_recent_weeks,
            n_forward_weeks=n_forward_weeks,
            signal_at_T=signal_summary,
            signal_direction=signal_direction,
            forward_positive=forward_positive,
            forward_negative=forward_negative,
            consistency_score=consistency_score,
            detail=detail_rows,
        )

    def rolling_backtest(
        self,
        entity: str,
        n_windows: int = 10,
        n_recent_weeks: int = 12,
        n_forward_weeks: int = 8,
        step_weeks: int = 4,
    ) -> dict:
        """
        滚动回测：从数据集中部往后，每隔 step_weeks 做一次单点回测。

        Parameters
        ----------
        n_windows    : 回测窗口总数
        step_weeks   : 每次向后移动多少周
        """
        g = self.graph
        eid = g.entity2id.get(entity)
        if eid is None:
            raise ValueError(f"找不到实体：{entity}")

        # 从数据集 1/3 处开始（确保有足够历史数据）
        start_t = g.max_time_id // 3
        results = []
        scores = []

        for i in range(n_windows):
            t_id = start_t + i * step_weeks
            if t_id + n_forward_weeks > g.max_time_id:
                break

            date_str = g.id2time.get(t_id, str(t_id))
            try:
                result = self.run(entity, date_str, n_recent_weeks, n_forward_weeks)
                results.append(result)
                scores.append(result.consistency_score)
                print(f"  [{i+1}/{n_windows}] {date_str}  方向={result.signal_direction:<8}  "
                      f"一致性={result.consistency_score:+.3f}")
            except Exception as e:
                print(f"  [{i+1}/{n_windows}] {date_str}  跳过：{e}")

        if not scores:
            return {"entity": entity, "error": "无有效回测结果"}

        avg_score = sum(scores) / len(scores)
        win_rate = sum(1 for s in scores if s > 0.2) / len(scores)

        report = {
            "entity": entity,
            "n_windows": len(results),
            "avg_consistency_score": round(avg_score, 3),
            "win_rate": round(win_rate, 3),   # 信号方向正确的比例
            "interpretation": (
                "KG 信号方向与后续事件高度一致" if avg_score > 0.3
                else "KG 信号有一定参考价值" if avg_score > 0
                else "KG 信号一致性较弱，建议结合其他指标"
            ),
        }

        print(f"\n{'='*50}")
        print(f"  滚动回测结果：{entity}")
        print(f"{'='*50}")
        print(f"  有效窗口数：{report['n_windows']}")
        print(f"  平均一致性得分：{report['avg_consistency_score']:+.3f}")
        print(f"  信号胜率：{report['win_rate']:.1%}")
        print(f"  解读：{report['interpretation']}")

        return report

    # ── 辅助方法 ─────────────────────────────────────────────────

    def _find_closest_time_id(self, date_str: str) -> Optional[int]:
        """在 time2id 中找最接近目标日期的 time_id"""
        # 直接匹配
        if date_str in self.graph.time2id:
            return self.graph.time2id[date_str]

        # 按字符串排序找最近的
        dates = sorted(self.graph.time2id.keys())
        for d in reversed(dates):
            if d <= date_str:
                return self.graph.time2id[d]
        return None

    def _compute_summary(self, eid: int, t_min: int, t_max: int) -> dict:
        """计算指定时间窗口内的信号摘要"""
        g = self.graph
        df = g.triples[
            (g.triples["t"] >= t_min) &
            (g.triples["t"] <= t_max) &
            ((g.triples["s"] == eid) | (g.triples["o"] == eid))
        ]
        rel_names = df["r"].map(g.id2relation)
        pos_count = int(rel_names.isin(self.POSITIVE_RELS).sum())
        neg_count = int(rel_names.isin(self.NEGATIVE_RELS).sum())
        return {
            "pos_count": pos_count,
            "neg_count": neg_count,
            "total": len(df),
        }

    def _judge_direction(self, pos: int, neg: int) -> str:
        if pos == 0 and neg == 0:
            return "neutral"
        ratio = (pos - neg) / (pos + neg)
        if ratio > 0.3:
            return "positive"
        if ratio < -0.3:
            return "negative"
        return "mixed"

    def _compute_consistency(
        self, direction: str, fwd_pos: int, fwd_neg: int
    ) -> float:
        """
        计算一致性得分 [-1, 1]：
        - direction=positive 且后续正面事件多 → 正分
        - direction=negative 且后续负面事件多 → 正分（方向一致）
        - 相反情况 → 负分
        """
        total_fwd = fwd_pos + fwd_neg
        if total_fwd == 0:
            return 0.0

        fwd_ratio = (fwd_pos - fwd_neg) / total_fwd  # [-1, 1]

        if direction == "positive":
            return round(fwd_ratio, 3)
        elif direction == "negative":
            return round(-fwd_ratio, 3)
        else:  # mixed / neutral
            return 0.0


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from kg_query import FinDKGGraph

    graph = FinDKGGraph()
    bt = Backtester(graph)

    print("\n=== 单点回测：Apple Inc. @ 2021-06-01 ===")
    result = bt.run("Apple Inc.", decision_date="2021-06-06",
                    n_recent_weeks=12, n_forward_weeks=8)
    print(result.summary())

    print("\n=== 滚动回测：Apple Inc. ===")
    report = bt.rolling_backtest("Apple Inc.", n_windows=8, step_weeks=4)
