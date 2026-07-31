"""
config.py
---------
全局配置。修改这里来调整数据路径、模型参数等，无需改动业务代码。

之前的问题（本次重构修复）：
  1. LLM_MODEL/LLM_MAX_TOKENS 定义了但从未被任何模块读取——
     llm_advisor.py / orchestrator.py / macro_agent.py 各自在
     构造函数默认参数里硬编码了一份模型名，改这里根本不生效。
  2. DATA_DIR / EMBEDDING_DIR 的"候选根目录"逻辑
     （knowledgeGraph/FinDKG 优先，同级 FinDKG 兜底）在 kg_query.py
     和 kg_predictor.py 里又各自重复实现了一遍。
  3. FEEDBACK_DB_PATH 定义了但 feedback_store.py 用的是自己的
     模块级 _DB_PATH，同样没有真正生效。

现在的约定：
  - 所有路径候选逻辑只在这里算一次，kg_query.py / kg_predictor.py /
    feedback_store.py 直接消费算好的结果，不再自己探测。
  - 每个 Agent 的 model / max_tokens 都有独立配置项，互不影响，
    可以按需要给 Critic 换更便宜的模型而不影响其他 Agent。
  - ANTHROPIC_API_KEY / FRED_API_KEY 的读取统一走本文件的两个
    get_xxx_api_key() 函数，其他模块不再各自 os.environ.get()。
"""

import os
from pathlib import Path


# ── 数据集根目录候选 ────────────────────────────────────────────────
# 优先尝试项目内部 Claude_capstone/knowledgeGraph/FinDKG/...
# （数据集已拷贝进项目，随项目一起分发，不再依赖外部同级目录）；
# 其次兼容旧的"同级摆放"布局：../knowledgeGraph/FinDKG/... 或 ../FinDKG/...
_KG_ROOT_CANDIDATES = [
    Path(__file__).parent / "knowledgeGraph" / "FinDKG",
    Path(__file__).parent.parent / "knowledgeGraph" / "FinDKG",
    Path(__file__).parent.parent / "FinDKG",
]


def _first_existing(candidates: list[Path]) -> Path:
    """返回第一个存在的候选路径；都不存在则返回第一个（用于生成清晰的报错信息）"""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# ── 数据路径 ──────────────────────────────────────────────────────
DATA_DIR = _first_existing(
    [root / "FinDKG_dataset" / "FinDKG-full" for root in _KG_ROOT_CANDIDATES]
)

# 从服务器导出的嵌入矩阵目录（static_structural.npy + dynamic_structural.npy）
EMBEDDING_DIR = _first_existing(
    [root / "embeddings" for root in _KG_ROOT_CANDIDATES]
)

# KGTransformer checkpoint（legacy，当前 kg_predictor.py 只依赖 EMBEDDING_DIR，
# 这里保留是为了以后如果要切回直接读 checkpoint 时有个统一入口）
CHECKPOINT_PATH = (
    _KG_ROOT_CANDIDATES[0] / "result" / "FinDKG" / "DKG"
    / "FinDKG_KGTransformer_overall_best_checkpoint_opt_edge.pt"
)


# ── LLM 参数（按 Agent 分别配置）──────────────────────────────────
# llm_advisor.AssetAdvisor（单 Agent 模式，main.py --entity 走这条路径）
ADVISOR_MODEL = "claude-haiku-4-5"
ADVISOR_MAX_TOKENS = 2048

# orchestrator.OrchestratorAgent 的 agentic loop（生成报告草稿）
ORCHESTRATOR_MODEL = "claude-haiku-4-5"
ORCHESTRATOR_MAX_TOKENS = 2048

# orchestrator.OrchestratorAgent 的 Critic 审查环节
# 独立配置，方便以后换成更便宜/更严格的模型而不影响草稿生成
CRITIC_MODEL = "claude-haiku-4-5"
CRITIC_MAX_TOKENS = 1024

# macro_agent.MacroAgent 宏观判断
MACRO_MODEL = "claude-haiku-4-5"
MACRO_MAX_TOKENS = 512


# ── 查询参数 ──────────────────────────────────────────────────────
DEFAULT_WEEKS = 12          # 默认查询最近多少周
MAX_EVENTS_IN_PROMPT = 8    # prompt 中每类事件最多显示几条


# ── 报告存储 ──────────────────────────────────────────────────────
REPORTS_DIR = Path(__file__).parent / "reports"   # Orchestrator 输出的 Markdown 报告目录

# ── 反馈存储 ──────────────────────────────────────────────────────
FEEDBACK_DB_PATH = Path(__file__).parent / "data" / "feedback.db"

# ── 回测参数 ──────────────────────────────────────────────────────
BACKTEST_FORWARD_WEEKS = 8   # 回测验证未来多少周
BACKTEST_N_WINDOWS = 10      # 滚动回测窗口数


# ── API Key 读取（集中一处）───────────────────────────────────────

def get_anthropic_api_key(explicit: str | None = None) -> str | None:
    """优先级：显式传入 > 环境变量 ANTHROPIC_API_KEY"""
    return explicit or os.environ.get("ANTHROPIC_API_KEY")


def get_fred_api_key(explicit: str | None = None) -> str | None:
    """优先级：显式传入 > 环境变量 FRED_API_KEY"""
    return explicit or os.environ.get("FRED_API_KEY")
