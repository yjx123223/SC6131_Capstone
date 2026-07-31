"""
mcp_servers/kg_server.py
------------------------
FinDKG 知识图谱 MCP Server

暴露两个工具：
  - query_kg_signals : 查询实体历史 KG 事件信号
  - get_feedback_stats: 查询历史建议评分统计

实际业务逻辑在 tools/kg_tools.py 和 tools/feedback_tools.py 中实现，
与 orchestrator.py 的 agentic loop 共用同一份代码，这里只是薄封装：
接收 MCP 调用参数 → 调用 tools 层函数 → 序列化成 JSON 字符串返回。

启动方式（stdio transport，供 MCP client 连接）：
  python mcp_servers/kg_server.py

在 Claude Desktop 注册（~/.claude/claude_desktop_config.json）：
  {
    "mcpServers": {
      "findkg": {
        "command": "python",
        "args": ["/绝对路径/Claude_capstone/mcp_servers/kg_server.py"]
      }
    }
  }
"""

import sys
import json
from pathlib import Path

# 将父目录加入 sys.path，使得 tools / kg_query / feedback_store 可导入
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

from tools import kg_tools, feedback_tools

# ── MCP Server 定义 ──────────────────────────────────────────────
mcp = FastMCP(
    "findkg",
    instructions=(
        "FinDKG 金融知识图谱工具。"
        "数据来源：华尔街日报新闻，时间跨度 2018-2023，13645 个金融实体，15 种关系类型。"
    ),
)


@mcp.tool()
def query_kg_signals(entity: str, weeks: int = 12) -> str:
    """
    查询目标实体在 FinDKG 知识图谱中的历史事件信号。

    返回：
    - 正面影响事件（Positive_Impact_On / Raise / Invests_In）
    - 负面影响事件（Negative_Impact_On / Decrease）
    - 其他关联事件

    Parameters
    ----------
    entity : 实体名称，如 "Apple Inc."、"Goldman Sachs Group"
    weeks  : 查询最近 N 周，默认 12
    """
    # 不显式传 graph，使用 tools.resources 的共享单例
    result = kg_tools.query_kg_signals(entity, weeks=weeks)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def get_feedback_stats(relation_type: str = "") -> str:
    """
    查询历史建议的用户评分统计。

    了解哪类 KG 关系类型（如 Positive_Impact_On、Invests_In）
    在过去生成的投资建议中平均评分更高，可用于调整信号权重。

    Parameters
    ----------
    relation_type : 按关系类型过滤（留空则返回全部统计）
    """
    result = feedback_tools.get_feedback_stats(relation_type=relation_type)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
