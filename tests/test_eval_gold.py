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


# ── AI 草稿的推导规则 ───────────────────────────────────────────

from eval.gold import DRAFT_PREFIX, apply_draft, article_keys, derive_row_label, review_diff


def _row(**kw):
    base = {"id": "1", "company": "AAPL", "source_company": "TSM", "hop": "1",
            "relation": "供应商", "url": "https://n/1", "material": "", "impact_on_target": "", "note": ""}
    base.update(kw)
    return base


def test_non_event_is_zero():
    assert derive_row_label(_row(), {"event": 0, "note": "股价播报"}) == ("0", "", "股价播报")


def test_target_own_event_uses_its_polarity():
    row = _row(source_company="AAPL", hop="0", relation="目标公司")
    assert derive_row_label(row, {"event": 1, "polarity": "positive"})[:2] == ("1", "positive")


def test_supplier_event_keeps_direction():
    label = {"event": 1, "polarity": "negative", "transmits": 1}
    assert derive_row_label(_row(), label)[:2] == ("1", "negative")


def test_competitor_event_is_inverted():
    row = _row(source_company="GOOGL", relation="竞争对手")
    label = {"event": 1, "polarity": "positive", "transmits": 1}
    assert derive_row_label(row, label)[:2] == ("1", "negative")


def test_industry_wide_event_is_not_inverted():
    row = _row(source_company="INTC", relation="竞争对手")
    label = {"event": 1, "polarity": "negative", "transmits": 1, "industry_wide": 1, "downstream": "negative"}
    assert derive_row_label(row, label)[:2] == ("1", "negative")


def test_downstream_override():
    label = {"event": 1, "polarity": "negative", "transmits": 1, "downstream": "neutral"}
    assert derive_row_label(_row(), label)[:2] == ("1", "neutral")


def test_event_without_transmission_is_zero():
    material, impact, note = derive_row_label(_row(), {"event": 1, "polarity": "positive", "transmits": 0})
    assert (material, impact) == ("0", "")
    assert "传导通路不明确" in note


def test_article_about_other_company():
    label = {"event": 1, "polarity": "negative", "about": "TSLA", "transmits": 1}
    assert derive_row_label(_row(), label)[0] == "0"                      # 目标是 AAPL
    assert derive_row_label(_row(company="TSLA"), label)[:2] == ("1", "negative")   # 目标就是主角


def test_duplicate_article_is_zero():
    label = {"event": 1, "polarity": "negative", "transmits": 1, "dup_of": "NFLX-1", "note": "同一次下调"}
    assert derive_row_label(_row(), label)[0] == "0"


def test_same_url_twice_under_one_target_counts_once():
    label = {"event": 1, "polarity": "negative", "transmits": 1}
    seen = set()
    assert derive_row_label(_row(), label, seen)[0] == "1"
    seen.add("https://n/1")
    material, _, note = derive_row_label(_row(source_company="AVGO"), label, seen)
    assert material == "0" and "重复" in note


# ── 应用草稿与修正率 ────────────────────────────────────────────

def test_apply_draft_fills_rows_and_marks_notes():
    store = _store()
    rows = build_rows(store)
    keys = article_keys(store)
    labels = {k: {"event": 0, "note": "噪音"} for k in keys.values()}
    # 让台积电最新一条成为可传导的负面事件
    tsm_key = keys[("TSM", store.articles("TSM")[-1]["url"])] if False else None
    newest_tsm = sorted(store.articles("TSM"), key=lambda a: a["published_at"], reverse=True)[0]
    labels[keys[("TSM", newest_tsm["url"])]] = {"event": 1, "polarity": "negative",
                                                "transmits": 1, "note": "产能受限"}

    rows, stats = apply_draft(rows, store, labels)

    assert stats["filled"] == len(rows) and stats["missing_label"] == 0
    assert stats["material"] == 1
    hit = [r for r in rows if r["material"] == "1"][0]
    assert hit["source_company"] == "TSM" and hit["impact_on_target"] == "negative"
    assert hit["note"] == DRAFT_PREFIX + "产能受限"
    assert all(r["note"].startswith(DRAFT_PREFIX) for r in rows)


def test_apply_draft_does_not_overwrite_human_labels():
    store = _store()
    rows = build_rows(store)
    rows[0]["material"] = "1"
    rows[0]["impact_on_target"] = "positive"
    rows[0]["note"] = "人工"
    labels = {k: {"event": 0} for k in article_keys(store).values()}

    rows, stats = apply_draft(rows, store, labels)
    assert stats["skipped_existing"] == 1
    assert rows[0]["note"] == "人工"


def test_project_article_labels_cover_the_real_pool():
    """项目里的 AI 草稿应覆盖真实快照的全部候选文章"""
    from eval.gold import load_article_labels
    from eval.snapshot import SnapshotStore
    import pathlib

    snap = pathlib.Path("eval/snapshots/2026-09-19/snapshot.json")
    if not snap.exists():
        pytest.skip("没有快照")
    store = SnapshotStore.load(snap)
    rows = build_rows(store)
    keys = article_keys(store)
    labels = load_article_labels()
    missing = {keys[(r["source_company"], r["url"])] for r in rows
               if keys.get((r["source_company"], r["url"])) not in labels}
    assert not missing, f"以下文章缺少判断：{sorted(missing)[:10]}"


def test_review_diff_counts_changes():
    draft = [_row(id="1", material="1", impact_on_target="negative"),
             _row(id="2", material="0", impact_on_target=""),
             _row(id="3", material="1", impact_on_target="positive")]
    reviewed = [_row(id="1", material="0", impact_on_target=""),
                _row(id="2", material="0", impact_on_target=""),
                _row(id="3", material="1", impact_on_target="negative")]

    result = review_diff(draft, reviewed)
    assert result["compared"] == 3
    assert result["changed_material"] == 1 and result["changed_impact"] == 2
    assert result["material_change_rate"] == 0.333
    assert result["any_change_rate"] == 0.667
    assert [c["id"] for c in result["changes"]] == ["1", "3"]
