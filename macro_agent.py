"""
macro_agent.py 好像现在用不到了，现在是用tool
--------------
Macro Agent：从 FRED 拉取宏观经济指标，调用 Claude 生成宏观环境判断。

输出格式（MacroSignal dict）：
{
    "direction":   "bullish" | "bearish" | "neutral",
    "confidence":  "high" | "medium" | "low",
    "rationale":   str,          # 一段简短的宏观判断理由
    "indicators":  dict,         # 原始指标快照
    "summary_text": str,         # 格式化后的指标文本（供 Orchestrator 参考）
}

依赖：
  pip install fredapi anthropic
FRED API Key 申请：https://fred.stlouisfed.org/docs/api/api_key.html
设置环境变量：export FRED_API_KEY='your-key'
"""

import json
from typing import Optional
import anthropic

import config


# FRED 数据系列定义
_FRED_SERIES = {
    "fed_funds_rate":   ("FEDFUNDS",   "联邦基金利率 (%)"),
    "treasury_10y":     ("DGS10",      "10年期国债收益率 (%)"),
    "cpi_yoy":          ("CPIAUCSL",   "CPI 同比变化 (%)"),
    "unemployment":     ("UNRATE",     "失业率 (%)"),
    "vix":              ("VIXCLS",     "VIX 恐慌指数"),
}


def _fetch_fred_indicators(fred_api_key: str) -> dict:
    """
    使用 fredapi 拉取最新宏观指标。
    返回 {key: {"value": float, "date": str, "label": str}} 的字典。
    """
    # macOS SSL 证书修复（Python 官方安装包不自带系统证书链）
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "缺少 fredapi 库。请运行：pip install fredapi"
        )

    fred = Fred(api_key=fred_api_key)
    result = {}

    for key, (series_id, label) in _FRED_SERIES.items():
        try:
            series = fred.get_series(series_id)
            series = series.dropna()
            if series.empty:
                result[key] = {"value": None, "date": "N/A", "label": label}
            else:
                latest_date = series.index[-1]
                latest_val  = float(series.iloc[-1])
                # 同比/环比变化（CPI 特殊处理为同比）
                prev_val = None
                if key == "cpi_yoy" and len(series) >= 13:
                    prev_val = float(series.iloc[-13])
                    latest_val = round((latest_val / prev_val - 1) * 100, 2)
                result[key] = {
                    "value": round(latest_val, 2),
                    "date":  latest_date.strftime("%Y-%m-%d"),
                    "label": label,
                }
        except Exception as e:
            result[key] = {"value": None, "date": "N/A", "label": label, "error": str(e)}

    return result


def _format_indicators(indicators: dict) -> str:
    """将指标 dict 格式化为可读文本块"""
    lines = []
    for key, info in indicators.items():
        val = info.get("value")
        date = info.get("date", "N/A")
        label = info.get("label", key)
        if val is not None:
            lines.append(f"  {label}：{val}  （{date}）")
        else:
            err = info.get("error", "数据不可用")
            lines.append(f"  {label}：N/A  （{err}）")
    return "\n".join(lines)


class MacroAgent:
    """
    宏观环境判断 Agent。

    使用示例
    --------
    >>> agent = MacroAgent()
    >>> signal = agent.analyze()
    >>> print(signal["direction"])      # "bearish" / "bullish" / "neutral"
    >>> print(signal["rationale"])
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        model: str = config.MACRO_MODEL,
        max_tokens: int = config.MACRO_MAX_TOKENS,
    ):
        """
        Parameters
        ----------
        anthropic_api_key : Anthropic API Key（未传则读 ANTHROPIC_API_KEY 环境变量）
        fred_api_key      : FRED API Key（未传则读 FRED_API_KEY 环境变量）
        model             : 使用的 Claude 模型，默认取 config.MACRO_MODEL
        max_tokens        : 回复最大 token 数，默认取 config.MACRO_MAX_TOKENS
        """
        ant_key = config.get_anthropic_api_key(anthropic_api_key)
        if not ant_key:
            raise ValueError(
                "未找到 Anthropic API Key。\n"
                "请设置：export ANTHROPIC_API_KEY='your-key'"
            )
        self.client = anthropic.Anthropic(api_key=ant_key)
        self.model = model
        self.max_tokens = max_tokens

        self.fred_api_key = config.get_fred_api_key(fred_api_key)
        if not self.fred_api_key:
            raise ValueError(
                "未找到 FRED API Key。\n"
                "请申请：https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "然后设置：export FRED_API_KEY='your-key'"
            )

    # ── 核心分析 ────────────────────────────────────────────────────

    def analyze(self) -> dict:
        """
        拉取宏观指标 → 调用 Claude 判断宏观方向 → 返回 MacroSignal dict

        Returns
        -------
        {
            "direction":    "bullish" | "bearish" | "neutral",
            "confidence":   "high" | "medium" | "low",
            "rationale":    str,
            "indicators":   dict,
            "summary_text": str,
        }
        """
        print("[MacroAgent] 拉取 FRED 宏观指标...")
        indicators = _fetch_fred_indicators(self.fred_api_key)
        summary_text = _format_indicators(indicators)

        print(f"[MacroAgent] 调用 {self.model} 进行宏观判断...")
        system_prompt = """你是一位宏观经济分析师。
你将收到一组最新的美国宏观经济指标，请基于这些数据对当前宏观环境做出判断。

输出要求（严格按 JSON 格式，不要输出其他内容）：
{
  "direction": "bullish" 或 "bearish" 或 "neutral",
  "confidence": "high" 或 "medium" 或 "low",
  "rationale": "不超过100字的判断理由（中文）"
}

判断依据参考：
- 利率高且持续上升 + 通胀高 → bearish
- 利率开始下降 + 通胀回落 + VIX 低 → bullish
- 信号混合或不明确 → neutral"""

        user_prompt = f"""以下是最新宏观经济指标：

{summary_text}

请根据以上数据，输出宏观环境判断（JSON 格式）。"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text.strip()

        # 解析 JSON，容错处理
        try:
            # 提取 JSON 块（防止 Claude 输出多余文字）
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            parsed = json.loads(raw_text[start:end])
            direction  = parsed.get("direction", "neutral")
            confidence = parsed.get("confidence", "low")
            rationale  = parsed.get("rationale", raw_text)
        except (json.JSONDecodeError, ValueError):
            # 解析失败时降级为 neutral
            direction  = "neutral"
            confidence = "low"
            rationale  = raw_text

        signal = {
            "direction":    direction,
            "confidence":   confidence,
            "rationale":    rationale,
            "indicators":   indicators,
            "summary_text": summary_text,
        }

        direction_zh = {"bullish": "偏多 📈", "bearish": "偏空 📉", "neutral": "中性 ⚪"}.get(direction, direction)
        print(f"[MacroAgent] 宏观判断：{direction_zh}（置信度：{confidence}）")
        print(f"[MacroAgent] 理由：{rationale}")

        return signal


# ── 快速测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MacroAgent()
    signal = agent.analyze()
    print("\n=== MacroSignal ===")
    print(f"方向：{signal['direction']}")
    print(f"置信度：{signal['confidence']}")
    print(f"理由：{signal['rationale']}")
    print(f"\n指标快照：\n{signal['summary_text']}")
