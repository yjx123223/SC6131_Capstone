"""
tests/test_eval_gold.py
------------------------
eval.gold：候选池生成口径、标注文件读写、校验与自一致率。
"""

import pytest

import config
from eval.gold import COLUMNS, build_rows, check, consistency, read_csv, write_csv
from eval.snapshot import SnapshotStore
from tests.test_eval_snapshot import _fake_fns


def _store(n_articles=9):
    def news(ticker, max_items=8):
        return {"ticker": ticker, "lookback_days": 14, "article_count": n_articles,
                "articles": [{"title": f"{ticker} 新闻 {i}", "publisher": "Reuters",
                              "published_at": f"2026-09-{10 + i:02d}T08:00+00:00",
                              "summary": "x" * 300, "url": f"https://n/{ticker}/{i}"}
                             for i in range(n_articles)]}

    fns = _fake_fns()
    fns["news_fn"] = news
    return SnapshotStore.record(["AAPL"], on_progress=lambda m: None, **fns)


@pytest.fixture
def rows():
    return build_rows(_store())


def test_pool_size_follows_visible_budget(rows):
    """目标公司 NEWS_MAX_ITEMS 条，每个邻居 KG_NEWS_PER_COMPANY 条"""
    store = _store()
    n_neighbors = len(store.neighbors("AAPL"))
    assert len(rows) == config.NEWS_MAX_ITEMS + n_neighbors * config.KG_NEWS_PER_COMPANY

    per_source = {}
    for r in rows:
        per_source[r["source_company"]] = per_source.get(r["source_company"], 0) + 1
    assert per_source["AAPL"] == config.NEWS_MAX_ITEMS
    assert per_source["TSM"] == config.KG_NEWS_PER_COMPANY


def test_row_fields_and_annotator_columns(rows):
    assert list(rows[0]) == COLUMNS
    first = rows[0]
    assert first["id"] == 1 and first["company"] == "AAPL"
    assert first["source_company"] == "AAPL" and first["hop"] == 0 and first["relation"] == "目标公司"
    assert first["published_at"] == "2026-09-18"          # 最新的排最前
    assert len(first["summary"]) == 200                    # 摘要截断
    assert (first["material"], first["impact_on_target"], first["note"]) == ("", "", "")
    assert [r["id"] for r in rows] == list(range(1, len(rows) + 1))


def test_relation_labels_include_hop2_path(rows):
    asml = [r for r in rows if r["source_company"] == "ASML"]
    assert asml and asml[0]["hop"] == 2
    assert asml[0]["relation"] == "上游供应商（经由 TSM）"
    tsm = [r for r in rows if r["source_company"] == "TSM"][0]
    assert tsm["relation"] == "供应商" and tsm["hop"] == 1


def test_rows_grouped_by_company_then_hop(rows):
    hops = [r["hop"] for r in rows]
    assert hops == sorted(hops), "同一目标公司内应按 hop 递增排列，便于标注时保持同一视角"


def test_csv_roundtrip(tmp_path, rows):
    path = write_csv(rows, tmp_path / "gold" / "gold_events.csv")
    loaded = read_csv(path)
    assert len(loaded) == len(rows)
    assert loaded[0]["title"] == rows[0]["title"]
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")   # BOM：Excel 打开不乱码


# ── 校验 ────────────────────────────────────────────────────────

def _labelled(**overrides):
    base = {"id": "1", "company": "AAPL", "hop": "1", "material": "1",
            "impact_on_target": "negative", "note": ""}
    base.update(overrides)
    return base


def test_check_counts_material_and_hops():
    result = check([
        _labelled(id="1", hop="0"),
        _labelled(id="2", hop="1"),
        _labelled(id="3", hop="2"),
        _labelled(id="4", material="0", impact_on_target=""),
        _labelled(id="5", company="MSFT", hop="1"),
    ])
    assert result["errors"] == []
    assert result["summary"]["labelled"] == 5
    assert result["summary"]["material"] == 4
    assert result["summary"]["material_hop1_plus"] == 3
    assert result["by_company"]["AAPL"]["material"] == 3


@pytest.mark.parametrize("row, message", [
    (_labelled(material="yes"), "material 只能是"),
    (_labelled(impact_on_target="bullish"), "impact_on_target 只能是"),
    (_labelled(impact_on_target=""), "必须填 impact_on_target"),
])
def test_check_reports_format_errors(row, message):
    assert any(message in e for e in check([row])["errors"])


def test_check_warns_on_unlabelled_and_empty_company():
    result = check([
        _labelled(id="1", company="AAPL", material="0", impact_on_target=""),
        _labelled(id="2", company="MSFT"),
        {"id": "3", "company": "MSFT", "hop": "1", "material": "", "impact_on_target": "", "note": ""},
    ])
    assert any("AAPL 没有任何 material=1" in w for w in result["warnings"])
    assert any("MSFT 只标了 1/2" in w for w in result["warnings"])


def test_check_warns_when_impact_set_on_non_material():
    result = check([_labelled(material="0", impact_on_target="negative")])
    assert any("将被忽略" in w for w in result["warnings"])


# ── 自一致率 ────────────────────────────────────────────────────

def test_consistency_compares_common_ids():
    first = [_labelled(id="1"), _labelled(id="2", material="0", impact_on_target=""),
             _labelled(id="3", impact_on_target="positive")]
    second = [_labelled(id="1"), _labelled(id="2"), _labelled(id="3", impact_on_target="negative"),
              _labelled(id="9")]

    result = consistency(first, second)
    assert result["compared"] == 3
    assert result["material_agreement"] == 0.667     # #2 的 material 变了（四舍五入到 3 位）
    assert result["full_agreement"] == 0.333         # #3 的方向也变了
    assert [d["id"] for d in result["disagreements"]] == ["2", "3"]


def test_consistency_without_overlap():
    assert consistency([_labelled(id="1")], [_labelled(id="2")])["compared"] == 0
