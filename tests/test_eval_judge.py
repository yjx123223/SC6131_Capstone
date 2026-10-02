"""eval/judge.py 的单元测试：假 client，不访问 API"""
import json
import types

import pytest

from eval import judge as judge_mod
from eval.judge import Judge, candidate_pool, claims_of


class _ToolUse:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class FakeClient:
    """按顺序吐出预设结果，并记录收到的 prompt"""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._payloads.pop(0)
        return types.SimpleNamespace(content=[_ToolUse(payload)], stop_reason="tool_use")


GOLD = [
    {"id": "1", "company": "AAPL", "source_company": "AAPL", "hop": "0", "relation": "目标公司",
     "published_at": "2026-09-18", "title": "Evercore 上调目标价", "summary": "至 380 美元", "material": "1"},
    {"id": "2", "company": "AAPL", "source_company": "QCOM", "hop": "1", "relation": "供应商",
     "published_at": "2026-09-17", "title": "高通份额下滑", "summary": "低于预期", "material": "1"},
    {"id": "3", "company": "MSFT", "source_company": "MSFT", "hop": "0", "relation": "目标公司",
     "published_at": "2026-09-16", "title": "别家的事件", "summary": "x", "material": "0"},
]


# ── 候选池 ──────────────────────────────────────────────────────

def test_candidate_pool_is_company_scoped():
    pool = candidate_pool(GOLD, "AAPL")
    assert [e["id"] for e in pool] == [1, 2]
    assert pool[1]["hop"] == 1 and pool[1]["source_company"] == "QCOM"


def test_candidate_pool_hides_material_label():
    """判定器不该看到哪些事件是重要的，否则等于泄题"""
    pool = candidate_pool(GOLD, "AAPL")
    assert all("material" not in e for e in pool)
    text = judge_mod._pool_text(pool)
    assert "material" not in text and "重要" not in text


def test_candidate_pool_truncates_long_summary():
    rows = [{"id": "1", "company": "AAPL", "source_company": "AAPL", "hop": "0",
             "relation": "目标公司", "published_at": "", "title": "t", "summary": "长" * 500}]
    assert len(candidate_pool(rows, "AAPL", summary_chars=50)[0]["summary"]) == 50


# ── 论断抽取 ────────────────────────────────────────────────────

def test_claims_splits_key_signals_and_risk_clauses():
    draft = {"key_signals": ["信号一", "信号二"],
             "risk_warnings": "主要风险：(1)估值偏高，PE 38.6 处历史高位；(2)供应链风险需要监控；(3)短"}
    claims = claims_of(draft)
    assert claims[:2] == ["信号一", "信号二"]
    assert any("估值偏高" in c for c in claims)
    assert any("供应链风险" in c for c in claims)
    assert all(len(c) > 8 for c in claims[2:])          # 太短的碎片被丢掉


def test_claims_handles_unsplittable_risk_text():
    claims = claims_of({"key_signals": [], "risk_warnings": "一段没有编号的较长风险描述文字"})
    assert claims == ["一段没有编号的较长风险描述文字"]


def test_claims_on_empty_draft():
    assert claims_of({}) == []


# ── 缓存 ────────────────────────────────────────────────────────

def _mention_payload(ids):
    return {"mentioned": [{"id": i, "evidence": f"原句{i}"} for i in ids]}


def test_judge_caches_by_prompt(tmp_path):
    client = FakeClient([_mention_payload([1])])
    j = Judge(client, cache_path=tmp_path / "cache.json")
    pool = candidate_pool(GOLD, "AAPL")

    first = j.mentions("报告正文", pool)
    second = j.mentions("报告正文", pool)

    assert first == second
    assert len(client.calls) == 1                      # 第二次命中缓存
    assert (j.hits, j.misses) == (1, 1)
    assert (tmp_path / "cache.json").exists()


def test_judge_cache_survives_restart(tmp_path):
    client = FakeClient([_mention_payload([2])])
    pool = candidate_pool(GOLD, "AAPL")
    Judge(client, cache_path=tmp_path / "cache.json").mentions("报告", pool)

    offline = Judge(None, cache_path=tmp_path / "cache.json")     # 没有 client
    assert offline.mentions("报告", pool)["mentioned"][0]["id"] == 2


def test_judge_offline_miss_raises(tmp_path):
    with pytest.raises(RuntimeError, match="离线"):
        Judge(None, cache_path=tmp_path / "cache.json").mentions("报告", candidate_pool(GOLD, "AAPL"))


def test_judge_cache_key_changes_with_prompt(tmp_path):
    client = FakeClient([_mention_payload([1]), _mention_payload([2])])
    j = Judge(client, cache_path=tmp_path / "cache.json")
    pool = candidate_pool(GOLD, "AAPL")
    j.mentions("报告甲", pool)
    j.mentions("报告乙", pool)
    assert len(client.calls) == 2


def test_judge_passes_temperature_zero(tmp_path):
    client = FakeClient([_mention_payload([])])
    Judge(client, cache_path=tmp_path / "c.json").mentions("报告", candidate_pool(GOLD, "AAPL"))
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["tool_choice"]["name"] == "emit_mentions"


def test_judge_raises_when_no_tool_use(tmp_path):
    class NoTool:
        messages = types.SimpleNamespace(
            create=lambda **kw: types.SimpleNamespace(content=[], stop_reason="end_turn"))
    with pytest.raises(RuntimeError, match="未返回"):
        Judge(NoTool(), cache_path=tmp_path / "c.json").mentions("报告", candidate_pool(GOLD, "AAPL"))


# ── 判定结果的清洗 ───────────────────────────────────────────────

def test_mentions_drops_ids_outside_pool(tmp_path):
    """模型偶尔会编出不存在的编号，不能让它们进指标"""
    client = FakeClient([{"mentioned": [{"id": 1, "evidence": "有"}, {"id": 999, "evidence": "编的"}]}])
    out = Judge(client, cache_path=tmp_path / "c.json").mentions("报告", candidate_pool(GOLD, "AAPL"))
    assert [m["id"] for m in out["mentioned"]] == [1]
    assert [m["id"] for m in out["dropped"]] == [999]


def test_citations_drops_out_of_range_index(tmp_path):
    client = FakeClient([{"verdicts": [
        {"index": 1, "verdict": "supported", "evidence": "数据里有"},
        {"index": 7, "verdict": "unsupported", "evidence": "越界"},
    ]}])
    j = Judge(client, cache_path=tmp_path / "c.json")
    out = j.citations(["论断一", "论断二"], "证据文本")
    assert [v["index"] for v in out["verdicts"]] == [1]
    assert out["claims"] == ["论断一", "论断二"]


# ── 批量判定 ────────────────────────────────────────────────────

RUN = {"variant": "C", "company": "AAPL", "repeat": 1, "git_commit": "abc1234",
       "report_md": "# 报告\nEvercore 给出 380 美元目标价",
       "draft": {"key_signals": ["Evercore 上调目标价至 380 美元"], "risk_warnings": ""},
       "tool_log": [{"tool": "query_market_data", "input": {},
                     "result": {"ticker": "AAPL", "technicals": {"last_close": 336.13},
                                "fundamentals": {}}}]}


def test_judge_all_writes_one_file_per_run(tmp_path):
    client = FakeClient([_mention_payload([1]),
                         {"verdicts": [{"index": 1, "verdict": "supported", "evidence": "e"}]}])
    j = Judge(client, cache_path=tmp_path / "c.json")

    results = judge_mod.judge_all([RUN], GOLD, j, out_dir=tmp_path / "j", on_progress=lambda s: None)

    path = judge_mod.judgement_path(tmp_path / "j", "C", "AAPL", 1)
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["mentions"]["mentioned"][0]["id"] == 1
    assert saved["citations"]["verdicts"][0]["verdict"] == "supported"
    assert saved["run_git_commit"] == "abc1234"
    assert results[0]["company"] == "AAPL"
    assert judge_mod.load_judgements(tmp_path / "j") == results


def test_judge_all_can_skip_citations(tmp_path):
    client = FakeClient([_mention_payload([])])
    j = Judge(client, cache_path=tmp_path / "c.json")
    out = judge_mod.judge_all([RUN], GOLD, j, out_dir=tmp_path / "j",
                              do_citations=False, on_progress=lambda s: None)
    assert "citations" not in out[0]
    assert len(client.calls) == 1


def test_judge_all_uses_full_evidence_text(tmp_path):
    """引用核验必须拿到补全后的证据摘要，否则又会把真数据判成无依据"""
    client = FakeClient([_mention_payload([]),
                         {"verdicts": [{"index": 1, "verdict": "supported", "evidence": "e"}]}])
    j = Judge(client, cache_path=tmp_path / "c.json")
    judge_mod.judge_all([RUN], GOLD, j, out_dir=tmp_path / "j", on_progress=lambda s: None)
    citation_prompt = client.calls[1]["messages"][0]["content"]
    assert "336.13" in citation_prompt                 # 工具数据进了核查 prompt


# ── 人工复核抽样 ─────────────────────────────────────────────────

def _judgement(variant, company, repeat, ids, verdicts):
    return {"variant": variant, "company": company, "repeat": repeat,
            "mentions": {"mentioned": [{"id": i, "evidence": f"e{i}"} for i in ids]},
            "citations": {"claims": ["论断一", "论断二"],
                          "verdicts": [{"index": i, "verdict": v, "evidence": "x"}
                                       for i, v in enumerate(verdicts, 1)]}}


def test_sample_for_review_is_deterministic_and_capped():
    js = [_judgement("A", "AAPL", 1, [1, 2], ["supported", "unsupported"]),
          _judgement("C", "AAPL", 1, [1], ["judgment", "supported"])]
    first = judge_mod.sample_for_review(js, GOLD, n=3)
    second = judge_mod.sample_for_review(js, GOLD, n=3)
    assert first == second and len(first) == 3
    assert {r["kind"] for r in first} <= {"mention", "citation"}
    assert all(r["human_verdict"] == "" for r in first)     # 留给人工填


def test_sample_csv_roundtrip(tmp_path):
    js = [_judgement("A", "AAPL", 1, [1], ["supported"])]
    rows = judge_mod.sample_for_review(js, GOLD, n=10)
    path = judge_mod.write_sample_csv(rows, tmp_path / "sample.csv")
    from eval.gold import read_csv
    back = read_csv(path)
    assert len(back) == len(rows)
    assert "human_verdict" in back[0]


# ── 模型返回格式异常的容错（实测 Haiku 会把数组返回成字符串）──────────

from eval.judge import coerce_items      # noqa: E402

# 取自真实缓存：mentioned 是一个 JSON 字符串，且里面的 evidence 含未转义引号
REAL_BAD_MENTIONS = '''[
  {
    "id": 3,
    "evidence": "新 CEO John Ternus 领导下推出 iPhone 18 Pro，获分析师认可，具有"Apple DNA"。"
  },
  {
    "id": 7,
    "evidence": "Evercore 上调目标价至 380 美元（9 月 18 日），显示机构看好。"
  }
]'''


def test_coerce_passes_through_well_formed_list():
    value = [{"id": 1, "evidence": "e"}]
    items, note = coerce_items(value, "mention")
    assert items == value and note == ""


def test_coerce_parses_stringified_array():
    items, note = coerce_items('[{"id": 5, "evidence": "原句"}]', "mention")
    assert items == [{"id": 5, "evidence": "原句"}]
    assert "字符串" in note


def test_coerce_repairs_invalid_inner_json():
    """真实故障样本：字符串化的数组 + 未转义的内嵌引号，json.loads 会失败"""
    import json as _json
    with pytest.raises(_json.JSONDecodeError):
        _json.loads(REAL_BAD_MENTIONS)

    items, note = coerce_items(REAL_BAD_MENTIONS, "mention")
    assert [m["id"] for m in items] == [3, 7]
    assert "Apple DNA" in items[0]["evidence"]          # 内嵌引号的证据也完整救回
    assert items[1]["evidence"].startswith("Evercore")
    assert "正则" in note


def test_coerce_repairs_verdicts():
    bad = '[{"index": 1, "verdict": "supported", "evidence": "数据里有 34.04%"}, ' \
          '{"index": 2, "verdict": "unsupported", "evidence": "找不到"}]'
    items, _ = coerce_items(bad.replace('"evidence": "数据里有', '"evidence": "数据"里"有'), "verdict")
    assert [v["index"] for v in items] == [1, 2]
    assert items[0]["verdict"] == "supported"


def test_coerce_falls_back_to_ids_only():
    """字段顺序都对不上时，至少把编号救回来——编号决定指标"""
    items, note = coerce_items('{"evidence": "先写证据", "id": 12}', "mention")
    assert items == [{"evidence": "先写证据", "id": 12}]   # dict 直接包一层

    items, note = coerce_items('乱七八糟 "id": 4 ... "id": 9 ...', "mention")
    assert [m["id"] for m in items] == [4, 9]
    assert all("未能解析" in m["evidence"] for m in items)


def test_coerce_handles_none_and_mixed_list():
    assert coerce_items(None, "mention")[0] == []
    items, note = coerce_items([{"id": 1, "evidence": "e"}, "垃圾"], "mention")
    assert items == [{"id": 1, "evidence": "e"}]
    assert "丢弃" in note


def test_mentions_marks_repaired_result(tmp_path):
    client = FakeClient([{"mentioned": REAL_BAD_MENTIONS}])
    pool = [{"id": 3, "source_company": "AAPL", "hop": 0, "relation": "", "published_at": "",
             "title": "t", "summary": "s"},
            {"id": 7, "source_company": "AAPL", "hop": 0, "relation": "", "published_at": "",
             "title": "t", "summary": "s"}]
    out = Judge(client, cache_path=tmp_path / "c.json").mentions("报告", pool)
    assert [m["id"] for m in out["mentioned"]] == [3, 7]
    assert "repaired" in out


def test_cached_bad_payload_is_repaired_on_read(tmp_path):
    """坏返回已经写进缓存也没关系：修复发生在读取时，不用清缓存重花钱"""
    client = FakeClient([{"mentioned": REAL_BAD_MENTIONS}])
    pool = [{"id": 3, "source_company": "A", "hop": 0, "relation": "", "published_at": "",
             "title": "", "summary": ""}]
    cache = tmp_path / "c.json"
    Judge(client, cache_path=cache).mentions("报告", pool)

    offline = Judge(None, cache_path=cache)
    assert [m["id"] for m in offline.mentions("报告", pool)["mentioned"]] == [3]


def test_judge_all_continues_after_one_failure(tmp_path, capsys):
    class Boom:
        def __init__(self):
            self.n = 0
            self.messages = types.SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("接口超时")
            return types.SimpleNamespace(content=[_ToolUse(_mention_payload([]))], stop_reason="tool_use")

    runs = [dict(RUN, company="AAPL", repeat=1), dict(RUN, company="AAPL", repeat=2)]
    j = Judge(Boom(), cache_path=tmp_path / "c.json")
    results = judge_mod.judge_all(runs, GOLD, j, out_dir=tmp_path / "j", do_citations=False)

    assert len(results) == 1 and results[0]["repeat"] == 2
    out = capsys.readouterr().out
    assert "判定失败" in out and "1 份判定失败" in out
    assert not judge_mod.judgement_path(tmp_path / "j", "C", "AAPL", 1).exists()


# ── 编号被返回成字符串（实测：一份判定因此抛 TypeError 中断）──────────

def test_coerce_converts_string_ids():
    """字符串编号在 mentions 里会被当成"编造的编号"悄悄丢掉，指标无声偏低"""
    items, note = coerce_items([{"id": "3", "evidence": "e"}, {"id": 7, "evidence": "e"}], "mention")
    assert [m["id"] for m in items] == [3, 7]
    assert "字符串" in note


def test_coerce_converts_string_index_in_verdicts():
    """实测故障：'<=' not supported between instances of 'int' and 'str'"""
    items, _ = coerce_items([{"index": "2", "verdict": "supported", "evidence": "e"}], "verdict")
    assert items[0]["index"] == 2
    assert isinstance(items[0]["index"], int)


def test_coerce_strips_event_id_prefix():
    items, _ = coerce_items([{"id": "E5", "evidence": "e"}, {"id": "[9]", "evidence": "e"}], "mention")
    assert [m["id"] for m in items] == [5, 9]


def test_coerce_drops_unparsable_ids():
    items, note = coerce_items([{"id": "第三条", "evidence": "e"}, {"id": 1, "evidence": "e"}], "mention")
    assert [m["id"] for m in items] == [1]
    assert "无法解析" in note


def test_coerce_drops_items_without_id():
    items, _ = coerce_items([{"evidence": "没有编号"}, {"id": 2, "evidence": "e"}], "mention")
    assert [m["id"] for m in items] == [2]


def test_citations_with_string_index_does_not_raise(tmp_path):
    """回归：B META r1 就是在这里中断的"""
    client = FakeClient([{"verdicts": [{"index": "1", "verdict": "supported", "evidence": "e"}]}])
    out = Judge(client, cache_path=tmp_path / "c.json").citations(["论断一"], "证据")
    assert out["verdicts"][0]["index"] == 1
