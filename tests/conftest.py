"""
tests/conftest.py
------------------
把项目根目录加入 sys.path，让测试能直接 `import orchestrator` /
`from tools import market_tools` 等，不需要 pip install 项目。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
