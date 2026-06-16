"""
feedback_store.py
-----------------
SQLite 反馈存储：记录每次建议会话，支持用户事后评分。

数据库自动创建于 Claude_capstone/data/feedback.db
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime


_DB_PATH = Path(__file__).parent / "data" / "feedback.db"


class FeedbackStore:
    """
    轻量级建议记录 + 用户反馈存储。

    使用示例
    --------
    >>> store = FeedbackStore()
    >>> session_id = store.log_advice("Apple Inc.", kg_summary, advice_text)
    >>> store.rate(session_id, rating=1, note="预测准确")
    >>> history = store.get_history("Apple Inc.", limit=5)
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS advice_sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity      TEXT    NOT NULL,
                    time_window INTEGER,
                    period      TEXT,
                    total_events INTEGER,
                    kg_summary  TEXT,          -- JSON
                    advice_text TEXT,
                    model       TEXT,
                    rating      INTEGER,       -- +1 / 0 / -1，NULL 表示未评分
                    note        TEXT,          -- 用户备注
                    created_at  REAL NOT NULL  -- UNIX timestamp
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity ON advice_sessions(entity)
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── 写入 ─────────────────────────────────────────────────────

    def log_advice(
        self,
        entity: str,
        kg_summary: dict,
        advice_text: str,
        model: str = "",
        time_window: int = 0,
    ) -> int:
        """
        记录一次建议会话，返回 session_id（供后续评分使用）。
        """
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO advice_sessions
                   (entity, time_window, period, total_events, kg_summary, advice_text, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity,
                    time_window,
                    kg_summary.get("period", ""),
                    kg_summary.get("total_events", 0),
                    json.dumps(kg_summary, ensure_ascii=False),
                    advice_text,
                    model,
                    time.time(),
                ),
            )
            return cur.lastrowid

    def rate(self, session_id: int, rating: int, note: str = ""):
        """
        对某次建议评分。

        Parameters
        ----------
        session_id : log_advice() 返回的 id
        rating     : +1（好）/ 0（中性）/ -1（差）
        note       : 可选备注
        """
        if rating not in (-1, 0, 1):
            raise ValueError("rating 只能是 +1 / 0 / -1")
        with self._conn() as conn:
            conn.execute(
                "UPDATE advice_sessions SET rating=?, note=? WHERE id=?",
                (rating, note, session_id),
            )
        print(f"[Feedback] 已记录评分 {'+' if rating > 0 else ''}{rating}（session #{session_id}）")

    # ── 查询 ─────────────────────────────────────────────────────

    def get_history(self, entity: Optional[str] = None, limit: int = 20) -> list[dict]:
        """
        查询历史建议记录。

        Parameters
        ----------
        entity : 过滤特定实体，None 表示全部
        limit  : 返回最近 N 条
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if entity:
                rows = conn.execute(
                    "SELECT * FROM advice_sessions WHERE entity=? ORDER BY created_at DESC LIMIT ?",
                    (entity, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM advice_sessions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            r["created_at"] = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
            r["kg_summary"] = json.loads(r["kg_summary"]) if r["kg_summary"] else {}
            results.append(r)
        return results

    def signal_accuracy_report(self) -> dict:
        """
        按关系类型统计：哪类 KG 信号与正/负反馈相关性更高。

        Returns
        -------
        {
          "total_rated": int,
          "positive_rate": float,     # 用户评 +1 的比例
          "signal_stats": {           # 各关系类型的平均评分
              "Positive_Impact_On": {"count": int, "avg_rating": float},
              ...
          }
        }
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT kg_summary, rating FROM advice_sessions WHERE rating IS NOT NULL"
            ).fetchall()

        if not rows:
            return {"total_rated": 0, "positive_rate": 0.0, "signal_stats": {}}

        total = len(rows)
        positive = sum(1 for _, r in rows if r == 1)
        signal_stats: dict[str, list[int]] = {}

        for kg_json, rating in rows:
            summary = json.loads(kg_json) if kg_json else {}
            # 统计出现的正面信号类型
            for event in summary.get("positive_impacts", []):
                rel = event.get("relation", "unknown")
                signal_stats.setdefault(rel, []).append(rating)
            # 统计出现的负面信号类型
            for event in summary.get("negative_impacts", []):
                rel = event.get("relation", "unknown")
                signal_stats.setdefault(rel, []).append(rating)

        stats_out = {}
        for rel, ratings in signal_stats.items():
            stats_out[rel] = {
                "count": len(ratings),
                "avg_rating": round(sum(ratings) / len(ratings), 3),
            }

        return {
            "total_rated": total,
            "positive_rate": round(positive / total, 3),
            "signal_stats": stats_out,
        }

    def print_history(self, entity: Optional[str] = None, limit: int = 10):
        """命令行打印历史记录"""
        records = self.get_history(entity, limit)
        if not records:
            print("暂无历史记录。")
            return
        print(f"\n{'='*60}")
        print(f"  历史建议记录（最近{limit}条）{'  实体: ' + entity if entity else ''}")
        print(f"{'='*60}")
        for r in records:
            rating_str = {1: "👍", 0: "😐", -1: "👎", None: "未评分"}.get(r["rating"], "?")
            print(f"\n[#{r['id']}] {r['created_at']}  {r['entity']}  {rating_str}")
            print(f"  时间段：{r['period']}  事件数：{r['total_events']}")
            # 只打印建议前3行
            preview = "\n  ".join(r["advice_text"].split("\n")[:3])
            print(f"  建议摘要：{preview}...")
            if r["note"]:
                print(f"  备注：{r['note']}")


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    store = FeedbackStore()
    print(f"数据库位置：{store.db_path}")

    # 模拟写入一条测试记录
    fake_summary = {
        "entity": "Apple Inc.",
        "period": "2022-10-23 ~ 2022-12-25",
        "total_events": 86,
        "positive_impacts": [{"subject": "Goldman Sachs", "relation": "Invests_In", "object": "Apple Inc.", "count": 1}],
        "negative_impacts": [{"subject": "Meta", "relation": "Decrease", "object": "Apple Inc.", "count": 1}],
    }
    sid = store.log_advice("Apple Inc.", fake_summary, "测试建议文本", model="claude-haiku-4-5")
    print(f"写入 session_id={sid}")

    store.rate(sid, rating=1, note="信号准确")
    store.print_history()
    print("\n信号准确率报告：", store.signal_accuracy_report())
