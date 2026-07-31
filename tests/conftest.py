"""
tests/conftest.py
------------------
让测试能直接 `import kg_query` / `from tools import kg_tools` 等，
不依赖项目被 pip install，只需把项目根目录加入 sys.path。
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def tiny_findkg_dir(tmp_path) -> Path:
    """
    构造一个最小可用的 FinDKG-full 格式数据集，供 kg_query.FinDKGGraph
    的测试使用，不依赖外部真实数据集。

    实体：Apple Inc.(0) / Goldman Sachs Group(1) / Meta Platforms(2)
    关系：Positive_Impact_On(0) / Negative_Impact_On(1) / Raise(2)
    时间：TimeID 0~2，对应 2022-01-03 / 2022-01-10 / 2022-01-17

    三元组（train.txt）：
      t=0: Goldman Sachs Group --Positive_Impact_On--> Apple Inc.
      t=1: Meta Platforms      --Negative_Impact_On--> Apple Inc.
      t=1: Goldman Sachs Group --Raise--> Apple Inc.
      t=2: Apple Inc.          --Positive_Impact_On--> Goldman Sachs Group
    """
    data_dir = tmp_path / "FinDKG-full"
    data_dir.mkdir()

    (data_dir / "entity2id.txt").write_text(
        "Apple Inc.\t0\t0\tORG\n"
        "Goldman Sachs Group\t1\t1\tORG\n"
        "Meta Platforms\t2\t2\tORG\n"
    )
    (data_dir / "relation2id.txt").write_text(
        "Positive_Impact_On\t0\n"
        "Negative_Impact_On\t1\n"
        "Raise\t2\n"
    )
    (data_dir / "time2id.txt").write_text(
        "TimeID,DATE_WK\n"
        "0,2022-01-03\n"
        "1,2022-01-10\n"
        "2,2022-01-17\n"
    )
    (data_dir / "train.txt").write_text(
        "1\t0\t0\t0\t0\n"
        "2\t1\t0\t1\t0\n"
        "1\t2\t0\t1\t0\n"
        "0\t0\t1\t2\t0\n"
    )
    # 不创建 valid.txt / test.txt：kg_query._load_split 对不存在的文件
    # 会返回空 DataFrame（空文件反而会让 pandas 抛 EmptyDataError）

    return data_dir
