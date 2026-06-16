"""
config.py
---------
全局配置。修改这里来调整数据路径、模型参数等，无需改动业务代码。
"""

from pathlib import Path

# ── 数据路径 ──────────────────────────────────────────────────────
# FinDKG-full 数据集位置（相对于本文件的路径）
# 优先尝试 ../knowledgeGraph/FinDKG/...，备用 ../FinDKG/...（同级目录结构）
_candidate1 = Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG" / "FinDKG_dataset" / "FinDKG-full"
_candidate2 = Path(__file__).parent.parent / "FinDKG" / "FinDKG_dataset" / "FinDKG-full"
DATA_DIR = _candidate1 if _candidate1.exists() else _candidate2

# ── LLM 参数 ──────────────────────────────────────────────────────
LLM_MODEL = "claude-opus-4-6"      # 或 "claude-sonnet-4-6"（更快更便宜）
LLM_MAX_TOKENS = 2048

# ── 查询参数 ──────────────────────────────────────────────────────
DEFAULT_WEEKS = 12          # 默认查询最近多少周
MAX_EVENTS_IN_PROMPT = 8    # prompt 中每类事件最多显示几条

# ── KGTransformer checkpoint ──────────────────────────────────────
# 训练完成后填入路径，例如：
#   CHECKPOINT_PATH = Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG" / "data" / "best_model.pt"
# 未填入（None）时预测功能自动禁用，不影响其他功能
CHECKPOINT_PATH = Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG" / "result" / "FinDKG" / "DKG" / "FinDKG_KGTransformer_overall_best_checkpoint_opt_edge.pt"

# 从服务器导出的嵌入矩阵目录（static_structural.npy + dynamic_structural.npy）
EMBEDDING_DIR = Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG" / "embeddings"

# ── 反馈存储 ──────────────────────────────────────────────────────
FEEDBACK_DB_PATH = Path(__file__).parent / "data" / "feedback.db"

# ── 回测参数 ──────────────────────────────────────────────────────
BACKTEST_FORWARD_WEEKS = 8   # 回测验证未来多少周
BACKTEST_N_WINDOWS = 10      # 滚动回测窗口数
