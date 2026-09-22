"""
tests/test_ticker_map.py
-------------------------
tools.ticker_map.resolve_ticker：公司名 / ticker → ticker 的解析规则。
"""

import pytest

from tools.ticker_map import resolve_ticker


@pytest.mark.parametrize("name, expected", [
    ("Apple Inc.", "AAPL"),
    ("apple", "AAPL"),
    ("Tesla, Inc.", "TSLA"),
    ("Microsoft Corporation", "MSFT"),
    ("Meta Platforms, Inc.", "META"),
    ("Amazon.com, Inc.", "AMZN"),
    ("Goldman Sachs Group", "GS"),
    ("JPMorgan Chase & Co.", "JPM"),
    ("Johnson & Johnson", "JNJ"),
    ("The Walt Disney Company", "DIS"),
    ("McDonald's Corporation", "MCD"),
])
def test_known_company_names_map_to_ticker(name, expected):
    result = resolve_ticker(name)
    assert result["ticker"] == expected
    assert result["matched_by"] == "mapping"
    assert result["entity"] == name


@pytest.mark.parametrize("raw", ["GE", "BRK-B", "BRK.B", "V"])
def test_uppercase_input_is_treated_as_ticker(raw):
    result = resolve_ticker(raw)
    assert result == {"ticker": raw, "entity": raw, "matched_by": "ticker"}


def test_uppercase_ticker_in_mapping_uses_mapping():
    # "META" 规范化后命中映射表，结果仍是 META
    assert resolve_ticker("META")["ticker"] == "META"


@pytest.mark.parametrize("raw", ["Unknown Startup Corp", "aapl", "NOTAREALTICKER", "12345"])
def test_unresolvable_input_returns_error_instead_of_guessing(raw):
    result = resolve_ticker(raw)
    assert "error" in result
    assert "ticker" not in result


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_input_returns_error(raw):
    assert "error" in resolve_ticker(raw)


@pytest.mark.parametrize("name, expected", [
    ("ASML Holding N.V.", "ASML"),
    ("Taiwan Semiconductor", "TSM"),
    ("Taiwan Semiconductor Manufacturing Company", "TSM"),
    ("Amazon Web Services", "AMZN"),
    ("Ford", "F"),
])
def test_news_style_company_names(name, expected):
    assert resolve_ticker(name)["ticker"] == expected
