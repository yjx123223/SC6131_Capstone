"""
kg_query.py
-----------
FinDKG-full 知识图谱查询模块
加载数据集，支持按实体名/时间窗口查询关系三元组

数据集路径的候选/探测逻辑统一在 config.py 的 DATA_DIR 里计算，
这里不再重复维护一份候选路径列表。
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class FinDKGGraph:
    """
    加载 FinDKG-full 数据集，提供实体/关系/时间三维查询能力。

    使用示例
    --------
    >>> g = FinDKGGraph()
    >>> results = g.query_entity("Apple Inc.", n_recent_weeks=8)
    >>> print(results)
    """

    def __init__(self, data_dir: Optional[str | Path] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            from config import DATA_DIR
            if not DATA_DIR.exists():
                raise FileNotFoundError(
                    f"找不到 FinDKG-full 数据集，请通过 data_dir 参数手动指定路径。\n"
                    f"config.py 中探测到的路径（不存在）：{DATA_DIR}"
                )
            self.data_dir = DATA_DIR
        self._load()

    # ── 数据加载 ─────────────────────────────────────────────────

    def _load(self):
        """加载所有映射表和三元组数据"""
        print(f"[FinDKG] 加载数据集：{self.data_dir}")

        # entity2id: 实体名 → (entity_id, entity_idx, entity_type)
        entity_df = pd.read_csv(
            self.data_dir / "entity2id.txt",
            sep="\t", header=None,
            names=["name", "id", "idx", "type"]
        )
        self.entity2id: dict[str, int] = dict(zip(entity_df["name"], entity_df["id"]))
        self.id2entity: dict[int, str] = dict(zip(entity_df["id"], entity_df["name"]))

        # relation2id: 关系名 → id
        rel_df = pd.read_csv(
            self.data_dir / "relation2id.txt",
            sep="\t", header=None,
            names=["name", "id"]
        )
        self.relation2id: dict[str, int] = dict(zip(rel_df["name"], rel_df["id"]))
        self.id2relation: dict[int, str] = dict(zip(rel_df["id"], rel_df["name"]))

        # time2id: time_id → 日期字符串
        time_df = pd.read_csv(self.data_dir / "time2id.txt")
        self.id2time: dict[int, str] = dict(zip(time_df["TimeID"], time_df["DATE_WK"]))
        self.time2id: dict[str, int] = dict(zip(time_df["DATE_WK"], time_df["TimeID"]))
        self.max_time_id: int = time_df["TimeID"].max()

        # 合并 train + valid + test 三元组
        def _load_split(fname):
            path = self.data_dir / fname
            if not path.exists():
                return pd.DataFrame(columns=["s", "r", "o", "t", "_"])
            return pd.read_csv(path, sep="\t", header=None,
                               names=["s", "r", "o", "t", "_"])

        self.triples = pd.concat([
            _load_split("train.txt"),
            _load_split("valid.txt"),
            _load_split("test.txt"),
        ], ignore_index=True)

        print(f"[FinDKG] 加载完成：{len(self.entity2id)} 实体，"
              f"{len(self.relation2id)} 关系，"
              f"{len(self.triples)} 三元组，"
              f"时间跨度 {self.id2time.get(0)} ~ {self.id2time.get(self.max_time_id)}")

    # ── 核心查询 ─────────────────────────────────────────────────

    def query_entity(
        self,
        entity_name: str,
        n_recent_weeks: int = 12,
        as_subject: bool = True,
        as_object: bool = True,
    ) -> pd.DataFrame:
        """
        查询某实体最近 n_recent_weeks 周内参与的所有三元组。

        Parameters
        ----------
        entity_name     : 实体名（需与 entity2id.txt 完全匹配）
        n_recent_weeks  : 查最近多少周（从最新时间戳往前数）
        as_subject      : 是否包含该实体作为主语的三元组
        as_object       : 是否包含该实体作为宾语的三元组

        Returns
        -------
        DataFrame，列: subject, relation, object, date
        """
        eid = self.entity2id.get(entity_name)
        if eid is None:
            # 模糊搜索提示
            candidates = self.fuzzy_search(entity_name, top_k=5)
            raise ValueError(
                f"找不到实体 '{entity_name}'。\n"
                f"相似实体（供参考）：{candidates}"
            )

        min_t = max(0, self.max_time_id - n_recent_weeks + 1)
        df = self.triples[self.triples["t"] >= min_t]

        masks = []
        if as_subject:
            masks.append(df["s"] == eid)
        if as_object:
            masks.append(df["o"] == eid)
        if not masks:
            return pd.DataFrame()

        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m

        result = df[combined].copy()
        result["subject"] = result["s"].map(self.id2entity)
        result["relation"] = result["r"].map(self.id2relation)
        result["object"] = result["o"].map(self.id2entity)
        result["date"] = result["t"].map(self.id2time)

        return result[["subject", "relation", "object", "date"]].sort_values("date", ascending=False)

    def query_between(self, entity_a: str, entity_b: str, n_recent_weeks: int = 26) -> pd.DataFrame:
        """查询两个实体之间的直接关系"""
        id_a = self.entity2id.get(entity_a)
        id_b = self.entity2id.get(entity_b)
        if id_a is None or id_b is None:
            missing = entity_a if id_a is None else entity_b
            raise ValueError(f"找不到实体：{missing}")

        min_t = max(0, self.max_time_id - n_recent_weeks + 1)
        df = self.triples[self.triples["t"] >= min_t]

        mask = ((df["s"] == id_a) & (df["o"] == id_b)) | \
               ((df["s"] == id_b) & (df["o"] == id_a))
        result = df[mask].copy()
        result["subject"] = result["s"].map(self.id2entity)
        result["relation"] = result["r"].map(self.id2relation)
        result["object"] = result["o"].map(self.id2entity)
        result["date"] = result["t"].map(self.id2time)

        return result[["subject", "relation", "object", "date"]].sort_values("date", ascending=False)

    def get_impact_summary(self, entity_name: str, n_recent_weeks: int = 12) -> dict:
        """
        聚合某实体的正/负面影响信号，返回结构化摘要，供 LLM prompt 使用。

        Returns
        -------
        {
            "entity": str,
            "period": str,
            "positive_impacts": [{"from/to": str, "direction": str, "count": int}],
            "negative_impacts": [...],
            "other_relations": [...],
            "total_events": int,
        }
        """
        df = self.query_entity(entity_name, n_recent_weeks=n_recent_weeks)
        if df.empty:
            return {"entity": entity_name, "total_events": 0, "period": f"最近{n_recent_weeks}周"}

        positive_rels = {"Positive_Impact_On", "Raise", "Invests_In"}
        negative_rels = {"Negative_Impact_On", "Decrease"}

        def _summarize(subset: pd.DataFrame) -> list[dict]:
            if subset.empty:
                return []
            counts = (
                subset.groupby(["subject", "relation", "object"])
                .size().reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            return counts.head(10).to_dict("records")

        pos_mask = df["relation"].isin(positive_rels)
        neg_mask = df["relation"].isin(negative_rels)
        other_mask = ~(pos_mask | neg_mask)

        period_start = df["date"].min()
        period_end = df["date"].max()

        return {
            "entity": entity_name,
            "period": f"{period_start} ~ {period_end}",
            "positive_impacts": _summarize(df[pos_mask]),
            "negative_impacts": _summarize(df[neg_mask]),
            "other_relations": _summarize(df[other_mask]),
            "total_events": len(df),
        }

    # ── 辅助工具 ─────────────────────────────────────────────────

    def fuzzy_search(self, keyword: str, top_k: int = 10) -> list[str]:
        """模糊搜索实体名（大小写不敏感子串匹配）"""
        kw = keyword.lower()
        return [name for name in self.entity2id if kw in name.lower()][:top_k]

    def list_relations(self) -> list[str]:
        return list(self.relation2id.keys())

    def stats(self) -> dict:
        return {
            "entities": len(self.entity2id),
            "relations": len(self.relation2id),
            "triples": len(self.triples),
            "time_range": f"{self.id2time.get(0)} ~ {self.id2time.get(self.max_time_id)}",
        }


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    g = FinDKGGraph()
    print("\n=== 统计 ===")
    print(g.stats())

    print("\n=== 搜索 'Apple' ===")
    print(g.fuzzy_search("Apple"))

    print("\n=== 查询 Apple Inc. 最近12周 ===")
    df = g.query_entity("Apple Inc.", n_recent_weeks=12)
    print(df.head(10).to_string(index=False))

    print("\n=== 影响摘要 ===")
    import json
    summary = g.get_impact_summary("Apple Inc.", n_recent_weeks=12)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
