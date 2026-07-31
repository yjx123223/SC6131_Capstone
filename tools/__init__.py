"""
tools
-----
共享工具层：query_kg_signals / query_macro / get_feedback_stats 的
唯一实现，供 orchestrator.py 的 agentic loop 和 mcp_servers/*.py
的 MCP tool 封装共同调用，避免同一段业务逻辑写两份。

各模块职责：
  resources.py       重资源（FinDKGGraph / FeedbackStore）
                      的懒加载单例，供不需要自带实例的调用方（如 MCP
                      server）使用。已经持有实例的调用方（如
                      orchestrator，实例来自 main.py）可以把实例显式
                      传入工具函数，不强制走这里的单例。
  kg_tools.py         query_kg_signals 工具实现
  macro_tools.py      query_macro 工具实现
  feedback_tools.py   get_feedback_stats 工具实现

约定：这一层的函数只返回普通 dict，成功/失败都不抛异常
（失败用 {"error": "..."}` 表示），是否要 json.dumps 序列化
由调用方（orchestrator 的 tool_result 组装 / MCP server 的
@mcp.tool() 封装）决定，不属于这一层的职责。
"""
