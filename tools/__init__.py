"""
tools
-----
共享工具层：Orchestrator agentic loop（以及 mcp_servers/*.py）调用的
工具的唯一实现，避免同一段业务逻辑写两份。

各模块职责：
  resources.py       重资源（FinDKGGraph / FeedbackStore）
                      的懒加载单例，供不需要自带实例的调用方（如 MCP
                      server）使用。已经持有实例的调用方（如
                      orchestrator，实例来自 main.py）可以把实例显式
                      传入工具函数，不强制走这里的单例。
  kg_tools.py         query_kg_signals 工具实现
  macro_tools.py      query_macro 工具实现
  feedback_tools.py   get_feedback_stats 工具实现

  实时数据工具（feat/market 新增，数据均为近期数据，带新鲜度校验）：
  market_tools.py     query_market_data：公司信息 / 估值财务指标 / 行情 + 技术指标
  news_tools.py       query_news：近期新闻（yfinance，免费无需 Key）
  sec_tools.py        query_sec_filings：近期 SEC 申报（EDGAR，免费需 User-Agent）
  indicators.py       技术指标纯函数（SMA / Wilder RSI / 波动率 / 涨跌幅）
  ticker_map.py       实体名 → ticker 解析
  yf_client.py        yfinance 懒加载与时间工具（便于测试注入）

  注：kg_tools.py 仍保留，但 Orchestrator 已不再调用（FinDKG 数据截止
  2023-01-01，与实时数据存在时间错位）。

约定：这一层的函数只返回普通 dict，成功/失败都不抛异常
（失败用 {"error": "..."}` 表示），是否要 json.dumps 序列化
由调用方（orchestrator 的 tool_result 组装 / MCP server 的
@mcp.tool() 封装）决定，不属于这一层的职责。
"""
