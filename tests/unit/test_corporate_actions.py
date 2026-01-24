from __future__ import annotations

import pandas as pd

from market_data.store import normalize_price_frame


def test_normalize_price_frame_defaults() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "close": [100.0, 101.0],
            "adj_close": [100.0, 101.0],
            "ticker": ["AAA", "AAA"],
        }
    )
    normalized = normalize_price_frame(prices, pd.DataFrame(), pd.DataFrame(), source="yahoo")
    assert "dividend" in normalized.columns
    assert "split_ratio" in normalized.columns
    assert normalized["dividend"].tolist() == [0.0, 0.0]
    assert normalized["split_ratio"].tolist() == [1.0, 1.0]


def test_normalize_price_frame_merges_actions() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "close": [100.0, 101.0],
            "adj_close": [100.0, 101.0],
            "ticker": ["AAA", "AAA"],
        }
    )
    dividends = pd.DataFrame({"date": ["2024-01-02"], "amount": [0.5]})
    splits = pd.DataFrame({"date": ["2024-01-01"], "ratio": [2.0]})
    normalized = normalize_price_frame(prices, dividends, splits, source="yahoo")
    assert normalized["dividend"].tolist() == [0.0, 0.5]
    assert normalized["split_ratio"].tolist() == [2.0, 1.0]
