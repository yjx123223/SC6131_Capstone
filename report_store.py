"""
report_store.py
-----------------
把渲染好的 Markdown 报告保存到本地文件。

从 orchestrator.py 的 OrchestratorAgent._save_report 拆分出来，
不含状态、不含网络调用，方便单独测试和复用。
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def save_report(entity: str, report_text: str, reports_dir: Optional[str | Path] = None) -> Path:
    """
    保存报告为 Markdown 文件。

    Parameters
    ----------
    entity      : 实体名，用于生成文件名
    report_text : 已渲染好的 Markdown 报告内容
    reports_dir : 保存目录，不传则使用 config.REPORTS_DIR

    Returns
    -------
    实际写入的文件路径
    """
    if reports_dir is not None:
        dir_path = Path(reports_dir)
    else:
        from config import REPORTS_DIR
        dir_path = Path(REPORTS_DIR)

    dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^\w\-]", "_", entity)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = dir_path / f"{safe_name}_{timestamp}.md"
    filepath.write_text(report_text, encoding="utf-8")
    print(f"[Orchestrator] 报告已保存：{filepath}")
    return filepath
