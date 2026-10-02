"""
feedback_store.py
-----------------
SQLite 反馈存储：记录每次生成的报告，支持用户事后评分。

数据库默认路径：config.FEEDBACK_DB_PATH（data/feedback.db）

表结构沿用早期版本（兼容已有的 feedback.db），列的当前含义：
  period        行情数据截至日期，如 "行情截至 2026-09-15"
  total_events  本次拿到的新闻条数
  kg_summary    本次建议依据的数据快照（JSON，见 OrchestratorAgent._extract_signal_snapshot）
  time_window   已不再使用（写入 0）
"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class FeedbackStore:
    """
    使用示例
    --------
    >>> store = FeedbackStore()
    >>> session_id = store.log_advice("Apple Inc.", snapshot, report_md)
    >>> store.rate(session_id, rating=1, note="判断准确")
    >>> history = store.get_history("Apple Inc.", limit=5)
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            from config import FEEDBACK_DB_PATH
            self.db_path = Path(FEEDBACK_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS advice_sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity       TEXT    NOT NULL,
                    time_window  INTEGER,
                    period       TEXT,
                    total_events INTEGER,
                    kg_summary   TEXT,          -- JSON 数据快照
                    advice_text  TEXT,
                    model        TEXT,
                    rating       INTEGER,       -- +1 / 0 / -1，NULL 表示未评分
                    note         TEXT,
                    created_at   REAL NOT NULL  -- UNIX timestamp
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON advice_sessions(entity)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── 写入 ─────────────────────────────────────────────────────

    def log_advice(self, entity: str, snapshot: dict, advice_text: str, model: str = "") -> int:
        """记录一次报告，返回 session_id（供后续评分使用）"""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO advice_sessions
                   (entity, time_window, period, total_events, kg_summary, advice_text, model, created_at)
                   VALUES (?, 0, ?, ?, ?, ?, ?, ?)""",
                (
                    entity,
                    snapshot.get("period", ""),
                    snapshot.get("total_events", 0),
                    json.dumps(snapshot, ensure_ascii=False),
                    advice_text,
                    model,
                    time.time(),
                ),
            )
            return cur.lastrowid

    def rate(self, session_id: int, rating: int, note: str = ""):
        """对某次报告评分：+1（好）/ 0（中性）/ -1（差）"""
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
        查询历史记录（按时间倒序）。返回的每条记录里，数据快照在 "snapshot" 字段。
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
            raw = r.pop("kg_summary")
            r["snapshot"] = json.loads(raw) if raw else {}
            r.pop("time_window", None)
            results.append(r)
        return results

    def print_history(self, entity: Optional[str] = None, limit: int = 10):
        """命令行打印历史记录"""
        records = self.get_history(entity, limit)
        if not records:
            print("暂无历史记录。")
            return
        print(f"\n{'='*60}")
        print(f"  历史报告记录（最近{limit}条）{'  实体: ' + entity if entity else ''}")
        print(f"{'='*60}")
        for r in records:
            rating_str = {1: "👍", 0: "😐", -1: "👎", None: "未评分"}.get(r["rating"], "?")
            snap = r["snapshot"]
            print(f"\n[#{r['id']}] {r['created_at']}  {r['entity']}  {rating_str}")
            if snap.get("recommendation"):
                print(f"  建议：{snap['recommendation']}（置信度 {snap.get('confidence')}）  "
                      f"{r['period'] or ''}  新闻 {r['total_events'] or 0} 条")
            else:
                print(f"  {r['period'] or ''}")   # 早期版本的记录没有这些字段
            if r["note"]:
                print(f"  备注：{r['note']}")
