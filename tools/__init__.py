"""
tools
-----
Orchestrator 可调用的工具实现。约定：函数只返回普通 dict，成功/失败都不抛异常
（失败用 {"error": "..."} 表示），序列化由调用方（OrchestratorLoop）负责。

  market_tools.py     query_market_data：公司信息 / 估值财务指标 / 行情 + 技术指标（yfinance）
  news_tools.py       query_news：近期新闻（yfinance，免费无需 Key）
  sec_tools.py        query_sec_filings：近期 SEC 申报（EDGAR，需 User-Agent）
  macro_tools.py      query_macro：宏观指标（FRED，需 FRED_API_KEY）
  kg_live_tools.py    query_company_graph：实时知识图谱与风险传导（见 live_kg/）

  indicators.py       技术指标纯函数（SMA / Wilder RSI / 波动率 / 涨跌幅）
  ticker_map.py       公司名 → ticker 解析
  yf_client.py        yfinance 懒加载与时间工具（便于测试注入）

所有实时数据工具都带新鲜度校验，数据过旧时返回 error。
"""
