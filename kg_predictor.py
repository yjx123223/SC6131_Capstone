"""
kg_predictor.py
---------------
KGTransformer 推断模块（DGL-free 版本）

不依赖 DGL，直接从导出的嵌入矩阵（static_structural.npy +
dynamic_structural.npy）计算实体表示，用余弦相似度对目标实体做
近似链接预测。

原理：
  拼接 static + dynamic 嵌入得到每个实体的综合表示，
  再计算目标实体与其余所有实体的余弦相似度，
  取 top-k 作为"预测的关联实体"。

  精度略低于完整 KGTransformer（缺少图结构信息），
  但对资产配置建议场景足够用，且无需 DGL/CUDA。

依赖：numpy（已安装）

注：早期版本还支持直接从训练 checkpoint（.pt）反序列化提取嵌入，
为此写了一套绕开 DGL/torchdata 依赖的 import mock 和安全 pickle
逻辑。现在统一改为消费导出好的 .npy 嵌入文件（config.EMBEDDING_DIR），
那套 checkpoint 相关的 hack 代码已不再被调用，随本次重构一并移除。
"""

import numpy as np
from pathlib import Path
from typing import Optional


class KGPredictor:
    """
    基于导出的 KGTransformer 实体嵌入做近似链接预测。

    使用示例
    --------
    >>> predictor = KGPredictor()
    >>> if predictor.available:
    ...     preds = predictor.predict_top_k("Apple Inc.", graph, top_k=10)
    ...     for p in preds: print(p)
    """

    def __init__(self, embedding_dir: Optional[str | Path] = None):
        self.available = False
        self._entity_emb = None   # shape: [n_entities, emb_dim]，L2归一化

        emb_dir = self._resolve_embedding_dir(embedding_dir)
        if emb_dir is None:
            print(
                "[KGPredictor] ⚠️  未找到嵌入文件目录，预测功能不可用。\n"
                "              请先在服务器上运行导出脚本，再 scp 传回本地，\n"
                "              或在 config.py 中设置 EMBEDDING_DIR"
            )
            return

        self._load_npy(emb_dir)

    # ── 初始化 ───────────────────────────────────────────────────

    def _resolve_embedding_dir(self, explicit) -> Optional[Path]:
        """
        按优先级查找嵌入目录：手动指定 > config.EMBEDDING_DIR。

        config.EMBEDDING_DIR 本身已经包含了
        "knowledgeGraph/FinDKG/embeddings 优先，同级 FinDKG/embeddings
        兜底"的候选逻辑（见 config.py），这里不再重复维护第二份候选列表。
        """
        if explicit:
            p = Path(explicit)
            if (p / "static_structural.npy").exists():
                return p

        try:
            from config import EMBEDDING_DIR
            p = Path(EMBEDDING_DIR)
            if (p / "static_structural.npy").exists():
                return p
        except Exception:
            pass

        return None

    def _load_npy(self, emb_dir: Path):
        """从 .npy 文件加载嵌入，完全不依赖 DGL/torchdata"""
        static_path  = emb_dir / "static_structural.npy"
        dynamic_path = emb_dir / "dynamic_structural.npy"

        print(f"[KGPredictor] 加载嵌入：{emb_dir}")
        try:
            s = np.load(static_path)   # [n_entities, static_dim]
            print(f"  static  shape: {s.shape}")
        except Exception as e:
            print(f"[KGPredictor] ⚠️  static_structural.npy 加载失败：{e}")
            return

        d = None
        if dynamic_path.exists():
            try:
                d = np.load(dynamic_path)   # [n_entities, n_rnn_layers, dynamic_dim]
                print(f"  dynamic shape: {d.shape}")
                # 取最后一个 RNN 层 → [n_entities, dynamic_dim]
                if d.ndim == 3:
                    d = d[:, -1, :]
                elif d.ndim == 4:
                    d = d[:, -1, :, 0]
            except Exception as e:
                print(f"[KGPredictor] ⚠️  dynamic_structural.npy 加载失败（跳过）：{e}")
                d = None

        # 拼接 static + dynamic
        combined = np.concatenate([s, d], axis=-1) if (d is not None and s.shape[0] == d.shape[0]) else s

        # L2 归一化
        norm = np.linalg.norm(combined, axis=-1, keepdims=True).clip(min=1e-8)
        self._entity_emb = (combined / norm).astype(np.float32)

        n, dim = self._entity_emb.shape
        print(f"[KGPredictor] ✅ 嵌入加载成功（{n} 个实体，维度 {dim}）")
        self.available = True

    # ── 核心预测 ─────────────────────────────────────────────────

    def predict_top_k(
        self,
        entity_name: str,
        graph,
        top_k: int = 10,
    ) -> list[dict]:
        """
        对目标实体，返回嵌入空间中最相似的 top-k 实体作为预测关联。

        Parameters
        ----------
        entity_name : 查询实体名
        graph       : FinDKGGraph 实例
        top_k       : 返回数量

        Returns
        -------
        [{"subject": str, "relation": str, "object": str, "score": float}, ...]
        """
        if not self.available:
            return []

        eid = graph.entity2id.get(entity_name)
        if eid is None or eid >= len(self._entity_emb):
            return []

        # 余弦相似度 = 点积（已归一化）
        query_vec = self._entity_emb[eid]           # [dim]
        scores = self._entity_emb @ query_vec       # [n_entities]

        # 排除自身，取 top-k
        scores[eid] = -np.inf
        topk_ids = np.argsort(scores)[::-1][:top_k]

        # 尝试推断最可能的关系类型（基于历史 KG 数据）
        rel_hint = self._infer_relation(eid, topk_ids, graph)

        results = []
        for obj_id in topk_ids:
            obj_name = graph.id2entity.get(int(obj_id), f"entity_{obj_id}")
            results.append({
                "subject": entity_name,
                "relation": rel_hint.get(int(obj_id), "Relate_To"),
                "object": obj_name,
                "score": float(scores[obj_id]),
            })
        return results

    def _infer_relation(self, src_id: int, candidate_ids, graph) -> dict[int, str]:
        """
        从历史三元组中查最近 8 周内 src→candidate 出现最频繁的关系，
        作为预测关系的参考标注。
        """
        hint = {}
        min_t = max(0, graph.max_time_id - 8)
        df = graph.triples[graph.triples["t"] >= min_t]
        mask = (df["s"] == src_id) & (df["o"].isin(candidate_ids))
        sub = df[mask]
        if not sub.empty:
            for obj_id, group in sub.groupby("o"):
                most_common_rel = group["r"].value_counts().idxmax()
                hint[int(obj_id)] = graph.id2relation.get(int(most_common_rel), "Relate_To")
        return hint

    # ── 格式化（供 llm_advisor 使用）────────────────────────────

    def format_predictions(self, predictions: list[dict]) -> str:
        if not predictions:
            return "（无预测数据）"
        return "\n".join(
            f"  - {p['subject']} --[{p['relation']}]--> {p['object']}  (相似度 {p['score']:.3f})"
            for p in predictions
        )

    # 兼容旧接口
    def predict_next(self, entity_name: str, graph, top_k: int = 10) -> list[dict]:
        return self.predict_top_k(entity_name, graph, top_k=top_k)


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = KGPredictor()
    print(f"预测模块可用：{predictor.available}")

    if predictor.available:
        from kg_query import FinDKGGraph
        graph = FinDKGGraph()
        print("\n预测 Apple Inc. 的关联实体（top 10）：")
        preds = predictor.predict_top_k("Apple Inc.", graph, top_k=10)
        for p in preds:
            print(f"  {p['subject']} --[{p['relation']}]--> {p['object']}  score={p['score']:.3f}")
