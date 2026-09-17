"""
tests/test_sec_tools.py
------------------------
tools.sec_tools.query_sec_filings：注入假的 http_get，不联网。
"""

from datetime import datetime, timezone

import pytest

from tools import sec_tools

NOW = datetime(2026, 9, 16, tzinfo=timezone.utc)
UA = "Test Runner test@example.com"

_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
}

_SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form":                  ["8-K", "4", "10-Q", "10-K", "10-Q"],
            "filingDate":            ["2026-08-01", "2026-08-02", "2026-07-31", "2025-10-31", "2025-05-02"],
            "reportDate":            ["2026-07-31", "", "2026-06-27", "2025-09-27", "2025-03-29"],
            "accessionNumber":       ["0000320193-26-000070", "0000320193-26-000071",
                                      "0000320193-26-000069", "0000320193-25-000079",
                                      "0000320193-25-000057"],
            "primaryDocument":       ["aapl-8k.htm", "form4.xml", "aapl-10q.htm", "aapl-10k.htm", "old.htm"],
            "primaryDocDescription": ["8-K", "", "10-Q", "10-K", "10-Q"],
        }
    },
}


class FakeHttp:
    def __init__(self, responses=None, exc_on=None):
        self.responses = responses or {}
        self.exc_on = exc_on
        self.calls = []

    def __call__(self, url, user_agent):
        self.calls.append((url, user_agent))
        if self.exc_on and self.exc_on in url:
            raise ConnectionError("SEC 403")
        for key, value in self.responses.items():
            if key in url:
                return value
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture(autouse=True)
def _clear_cache():
    sec_tools.clear_cache()
    yield
    sec_tools.clear_cache()


def _http():
    return FakeHttp({"company_tickers.json": _TICKERS, "CIK0000320193.json": _SUBMISSIONS})


def test_happy_path_filters_forms_and_lookback():
    http = _http()
    result = sec_tools.query_sec_filings("Apple Inc.", user_agent=UA, http_get=http, now=NOW)

    assert result["cik"] == 320193
    assert result["company"] == "Apple Inc."
    forms = [(f["form"], f["filing_date"]) for f in result["filings"]]
    # Form 4 不在默认类型里；2025-05-02 超出 365 天窗口
    assert forms == [("8-K", "2026-08-01"), ("10-Q", "2026-07-31"), ("10-K", "2025-10-31")]
    assert all(ua == UA for _, ua in http.calls)


def test_builds_real_archive_url():
    result = sec_tools.query_sec_filings("AAPL", user_agent=UA, http_get=_http(), now=NOW)
    assert result["filings"][0]["url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000070/aapl-8k.htm"
    )


def test_custom_form_types_and_max_items():
    result = sec_tools.query_sec_filings(
        "AAPL", form_types=["10-q", "10-K"], max_items=1, user_agent=UA, http_get=_http(), now=NOW
    )
    assert [f["form"] for f in result["filings"]] == ["10-Q"]


def test_missing_user_agent_returns_error_without_network(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    http = _http()
    result = sec_tools.query_sec_filings("AAPL", http_get=http, now=NOW)
    assert "SEC_EDGAR_USER_AGENT" in result["error"]
    assert http.calls == []


def test_user_agent_read_from_env(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Env Agent env@example.com")
    http = _http()
    sec_tools.query_sec_filings("AAPL", http_get=http, now=NOW)
    assert http.calls[0][1] == "Env Agent env@example.com"


def test_ticker_not_in_sec_returns_error():
    result = sec_tools.query_sec_filings("GE", user_agent=UA, http_get=_http(), now=NOW)
    assert "未找到" in result["error"]


def test_dot_ticker_matches_sec_dash_format():
    http = FakeHttp({
        "company_tickers.json": _TICKERS,
        "CIK0001067983.json": {"name": "BERKSHIRE", "filings": {"recent": {
            "form": ["10-Q"], "filingDate": ["2026-08-04"], "accessionNumber": ["0000950170-26-000001"],
            "primaryDocument": ["brk.htm"],
        }}},
    })
    result = sec_tools.query_sec_filings("BRK.B", user_agent=UA, http_get=http, now=NOW)
    assert result["cik"] == 1067983
    assert result["filings"][0]["report_date"] is None     # 缺列时不报错


def test_no_recent_filings_returns_error():
    result = sec_tools.query_sec_filings("AAPL", user_agent=UA, http_get=_http(),
                                         now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert set(result) == {"error"}


def test_ticker_list_failure_returns_error():
    http = FakeHttp(exc_on="company_tickers.json")
    result = sec_tools.query_sec_filings("AAPL", user_agent=UA, http_get=http, now=NOW)
    assert "SEC 403" in result["error"]


def test_submissions_failure_returns_error():
    http = FakeHttp({"company_tickers.json": _TICKERS}, exc_on="CIK0000320193")
    result = sec_tools.query_sec_filings("AAPL", user_agent=UA, http_get=http, now=NOW)
    assert "error" in result


def test_cik_list_is_cached_across_calls():
    http = _http()
    sec_tools.query_sec_filings("AAPL", user_agent=UA, http_get=http, now=NOW)
    sec_tools.query_sec_filings("AAPL", user_agent=UA, http_get=http, now=NOW)
    ticker_calls = [u for u, _ in http.calls if "company_tickers" in u]
    assert len(ticker_calls) == 1
