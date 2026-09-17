"""
config.py
---------
全局配置：模型参数、数据工具参数、知识图谱参数、存储路径、API Key 读取。
修改这里即可调整行为，无需改动业务代码。

约定：
  - 每个 Agent 的 model / max_tokens 独立配置，互不影响
  - ANTHROPIC_API_KEY / FRED_API_KEY / SEC_EDGAR_USER_AGENT 统一通过本文件的
    get_xxx() 读取，其他模块不直接调用 os.environ.get()
"""

import os
from pathlib import Path


# ── LLM 参数（按 Agent 分别配置）──────────────────────────────────
# OrchestratorLoop：agentic tool-use 循环与修订
# emit_report 有十多个字段且以中文为主，一次输出常超过 2k token，因此设为 8192
ORCHESTRATOR_MODEL = "claude-haiku-4-5"
ORCHESTRATOR_MAX_TOKENS = 8192

# CriticAgent：独立审查草稿
CRITIC_MODEL = "claude-haiku-4-5"
CRITIC_MAX_TOKENS = 2048


# ── 实时市场数据工具（tools/market_tools.py / news_tools.py / sec_tools.py）──
# 所有实时数据工具都做"新鲜度"校验：数据比阈值更旧时返回 error，
# 避免把过期数据当成"最新"喂给 Orchestrator。
MARKET_ALLOWED_PERIODS = ("1mo", "3mo", "6mo", "1y")   # 行情回看窗口上限 1 年
MARKET_DEFAULT_PERIOD = "3mo"
MARKET_MAX_STALENESS_DAYS = 7      # 最新一根 K 线距今超过 N 天视为过期（覆盖长周末/节假日）
MARKET_RECENT_CLOSES = 10          # 返回给 LLM 的最近收盘价条数

NEWS_LOOKBACK_DAYS = 14            # 只保留最近 N 天的新闻
NEWS_MAX_ITEMS = 8

SEC_LOOKBACK_DAYS = 365            # 只保留最近 N 天的 SEC 申报
SEC_MAX_FILINGS = 6
SEC_DEFAULT_FORMS = ("10-K", "10-Q", "8-K", "20-F", "6-K")
HTTP_TIMEOUT_SECONDS = 20


# ── 实时知识图谱（live_kg/ 与 tools/kg_live_tools.py）──────────────
KG_SEED_PATH = Path(__file__).parent / "live_kg" / "seed_relations.csv"   # 人工种子关系
KG_MAX_HOP1 = 5                # 1 跳邻居最多几家（供应商 > 客户 > 合作 > 竞争 的优先级截断）
KG_MAX_HOP2 = 3                # 2 跳（供应商的供应商）最多几家
KG_HOP_DECAY = {1: 1.0, 2: 0.5}   # 风险分数的跳数衰减
KG_NEWS_PER_COMPANY = 5        # 每家邻居公司拉取的新闻条数
KG_EXTRACT_MODEL = "claude-haiku-4-5"
KG_EXTRACT_MAX_TOKENS = 2048
KG_MAX_WORKERS = 6             # 邻居新闻拉取 / 事件抽取的并行线程数
KG_MAX_EVENTS_PER_COMPANY_IN_DIAGRAM = 3   # Mermaid 图中每家公司最多画几个事件


# ── 报告存储 ──────────────────────────────────────────────────────
REPORTS_DIR = Path(__file__).parent / "reports"   # Orchestrator 输出的 Markdown 报告目录

# ── 反馈存储 ──────────────────────────────────────────────────────
FEEDBACK_DB_PATH = Path(__file__).parent / "data" / "feedback.db"

# ── API Key 读取（集中一处）───────────────────────────────────────

def get_anthropic_api_key(explicit: str | None = None) -> str | None:
    """优先级：显式传入 > 环境变量 ANTHROPIC_API_KEY"""
    return explicit or os.environ.get("ANTHROPIC_API_KEY")


def get_fred_api_key(explicit: str | None = None) -> str | None:
    """优先级：显式传入 > 环境变量 FRED_API_KEY"""
    return explicit or os.environ.get("FRED_API_KEY")


def get_sec_user_agent(explicit: str | None = None) -> str | None:
    """
    优先级：显式传入 > 环境变量 SEC_EDGAR_USER_AGENT

    SEC 要求请求头 User-Agent 声明身份和联系邮箱，
    格式如 "Your Name your.email@example.com"，否则会返回 403。
    """
    return explicit or os.environ.get("SEC_EDGAR_USER_AGENT")
