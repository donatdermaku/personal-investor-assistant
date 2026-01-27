from __future__ import annotations

import pandas as pd

from market_data.rate_limiter import validate_price_cache
from market_data.store import normalize_price_frame


def test_price_frame_casts_float32() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "close": [100.0],
            "adj_close": [101.0],
        }
    )
    normalized = normalize_price_frame(prices, pd.DataFrame(), pd.DataFrame(), source="yahoo")
    assert str(normalized["close"].dtype) == "float32"
    assert str(normalized["adj_close"].dtype) == "float32"
    assert str(normalized["dividend"].dtype) == "float32"
    assert str(normalized["split_ratio"].dtype) == "float32"


def test_validate_price_cache_start_date_guard() -> None:
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "close": 1.0, "adj_close": 1.0})
    is_valid, reasons = validate_price_cache(
        df,
        required_start="2020-01-01",
        required_end="2022-01-10",
        min_rows=50,
    )
    assert is_valid is False
    assert any("START_NOT_COVERED" in reason for reason in reasons)


def test_validate_price_cache_duplicate_dates() -> None:
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates, "close": 1.0, "adj_close": 1.0})
    df.loc[1, "date"] = df.loc[0, "date"]
    is_valid, reasons = validate_price_cache(
        df,
        required_start="2022-01-01",
        required_end="2022-01-05",
        min_rows=1,
    )
    assert is_valid is False
    assert any("DUPLICATE_DATES" in reason for reason in reasons)
