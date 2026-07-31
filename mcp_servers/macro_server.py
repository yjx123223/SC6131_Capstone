"""
mcp_servers/macro_server.py
---------------------------
宏观经济数据 MCP Server

暴露一个工具：
  - query_macro : 从 FRED 拉取最新宏观指标

实际业务逻辑在 tools/macro_tools.py 中实现，与 orchestrator.py 的
agentic loop 共用同一份代码，这里只是薄封装：从环境变量取
FRED_API_KEY → 调用 tools 层函数 → 序列化成 JSON 字符串返回。

依赖环境变量：FRED_API_KEY
申请地址：https://fred.stlouisfed.org/docs/api/api_key.html

启动方式：
  FRED_API_KEY=your-key python mcp_servers/macro_server.py

在 Claude Desktop 注册（~/.claude/claude_desktop_config.json）：
  {
    "mcpServers": {
      "macro": {
        "command": "python",
        "args": ["/绝对路径/Claude_capstone/mcp_servers/macro_server.py"],
        "env": {
          "FRED_API_KEY": "your-fred-key"
        }
      }
    }
  }
"""

import sys
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

import config
from tools import macro_tools

mcp = FastMCP(
    "macro",
    instructions=(
        "美国宏观经济数据工具，数据来源：FRED（美联储经济数据库）。"
        "提供联邦基金利率、国债收益率、CPI、失业率、VIX 等实时指标。"
    ),
)


@mcp.tool()
def query_macro(indicators: list[str] | None = None) -> str:
    """
    从 FRED 获取最新美国宏观经济指标。

    可查询指标（indicators 参数可选，不传则返回全部）：
    - fed_funds_rate : 联邦基金利率（货币政策松紧）
    - treasury_10y   : 10年期国债收益率（长端利率）
    - cpi_yoy        : CPI 同比（通胀水平）
    - unemployment   : 失业率（劳动力市场）
    - vix            : VIX 恐慌指数（市场风险偏好）

    Parameters
    ----------
    indicators : 需要查询的指标列表，留空返回全部
    """
    fred_api_key = config.get_fred_api_key()
    result = macro_tools.query_macro(fred_api_key, indicators=indicators)
    return json.dumps(result, ensure_ascii=False, default=str)


if __name__ == "__main__":
    mcp.run()
