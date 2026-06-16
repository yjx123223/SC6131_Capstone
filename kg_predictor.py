"""
kg_predictor.py
---------------
KGTransformer 推断模块（DGL-free 版本）

不依赖 DGL，直接从 checkpoint 中提取实体嵌入，
用余弦相似度对目标实体做近似链接预测。

原理：
  checkpoint 里存有训练/验证完毕的实体嵌入向量
  (static_entity_emb + val_dynamic_entity_emb)。
  将二者拼接得到每个实体的综合表示，
  再计算目标实体与其余所有实体的余弦相似度，
  取 top-k 作为"预测的关联实体"。

  精度略低于完整 KGTransformer（缺少图结构信息），
  但对资产配置建议场景足够用，且无需 DGL/CUDA。

依赖：torch（已安装），numpy（已安装）
"""

import sys
import pickle
import types
import numpy as np
from pathlib import Path
from typing import Optional


def _install_import_mocks():
    """
    两步走：
    1. 把已安装但有缺失子模块的真实包（torchdata、dgl.graphbolt）
       从 sys.modules 中清除，换成空 mock，防止它们触发连锁 import。
    2. 安装 meta_path finder，拦截后续任何对这些包的 import。
    """
    import importlib.abc
    import importlib.machinery

    # 需要完全替换为 mock 的前缀（包括已安装但不完整的）
    MOCK_PREFIXES = ("torchdata", "dgl.graphbolt", "torch_scatter",
                     "setuptools.extern")

    def _make_mod(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__    = []
        mod.__package__ = name
        mod.__spec__    = None
        return mod

    # Step 1: 强制清除 sys.modules 中已存在的不完整包，替换为 mock
    for key in list(sys.modules.keys()):
        if any(key == p or key.startswith(p + ".") for p in MOCK_PREFIXES):
            sys.modules[key] = _make_mod(key)

    # Step 2: 安装 finder 拦截未来的 import
    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return _make_mod(spec.name)

        def exec_module(self, module):
            sys.modules[module.__name__] = module
            if "." in module.__name__:
                parent_name, attr = module.__name__.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent and not hasattr(parent, attr):
                    setattr(parent, attr, module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if any(fullname == p or fullname.startswith(p + ".")
                   for p in MOCK_PREFIXES):
                if fullname not in sys.modules:
                    return importlib.machinery.ModuleSpec(
                        fullname, _Loader(), is_package=True)
            return None

    if not any(type(f).__name__ == "_Finder" for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def _make_dummy(module_name: str = "", name: str = "Dummy"):
    """
    创建一个通用占位类：
    - 接受任意构造参数
    - __getattr__ 返回自身（可以无限链式调用，不会 NoneType 报错）
    - 支持迭代、长度查询
    - 支持 namedtuple 协议（_fields / _make）
    """
    def _getattr(self, k):
        # 返回一个 lambda，调用它也返回 self（避免 NoneType is not callable）
        return lambda *a, **kw: self

    cls = type(name, (), {
        "__init__":    lambda self, *a, **kw: None,
        "__repr__":    lambda self: f"<Dummy {module_name}.{name}>",
        "__iter__":    lambda self: iter([]),
        "__len__":     lambda self: 0,
        "__getattr__": _getattr,
        "__call__":    lambda self, *a, **kw: self,
        "_fields":     (),
        "_make":       classmethod(lambda cls, *a, **kw: cls()),
    })
    return cls


def _make_safe_pickle():
    """
    返回一个自定义 pickle 模块，其 Unpickler 在遇到无法导入的类时，
    自动用 Dummy 对象替代，而不是抛出 ImportError。
    供 torch.load(pickle_module=...) 使用。
    """

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module_name: str, name: str):
            try:
                return super().find_class(module_name, name)
            except (ImportError, AttributeError, ModuleNotFoundError):
                return _make_dummy(module_name, name)

    safe_pkl = types.ModuleType("_safe_pickle")
    safe_pkl.Unpickler        = _SafeUnpickler
    safe_pkl.load             = lambda f, **kw: _SafeUnpickler(f).load()
    safe_pkl.loads            = pickle.loads
    safe_pkl.UnpicklingError  = pickle.UnpicklingError
    safe_pkl.PicklingError    = pickle.PicklingError
    safe_pkl.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    return safe_pkl

_FINDKG_ROOT_CANDIDATES = [
    Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG",
    Path(__file__).parent.parent / "FinDKG",
]


def _find_findkg_root() -> Optional[Path]:
    for p in _FINDKG_ROOT_CANDIDATES:
        if (p / "DKG").exists():
            return p
    return None


class KGPredictor:
    """
    基于 KGTransformer checkpoint 的实体嵌入做近似链接预测。

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
        """按优先级查找嵌入目录：手动指定 > config.py > 默认路径"""
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

        # 默认路径
        for candidate in [
            Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG" / "embeddings",
            Path(__file__).parent.parent / "FinDKG" / "embeddings",
        ]:
            if (candidate / "static_structural.npy").exists():
                return candidate

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

    @staticmethod
    def _extract_tensor(emb_obj, attr: str):
        """
        从 MultiAspectEmbedding（namedtuple）中提取张量。
        MultiAspectEmbedding.structural 是 nn.Parameter（tensor 子类），
        可直接当 tensor 使用。
        """
        import torch
        if emb_obj is None:
            return None
        # 直接就是 tensor / Parameter
        if isinstance(emb_obj, torch.Tensor):
            return emb_obj.detach()
        # namedtuple / 有属性的对象
        if hasattr(emb_obj, attr):
            val = getattr(emb_obj, attr)
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val.detach()
            # nn.Embedding（兼容旧格式）
            if isinstance(val, torch.nn.Embedding):
                return val.weight.detach()
        # dict 格式
        if isinstance(emb_obj, dict) and attr in emb_obj:
            v = emb_obj[attr]
            if isinstance(v, torch.Tensor):
                return v.detach()
        return None

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
