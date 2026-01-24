import pandas as pd

from market_data.contracts import validate_price_frame, MarketDataError
from market_data.yahoo import _normalize_prices


def test_normalize_prices_has_date_column():
    raw = pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})
    normalized = _normalize_prices(raw)
    assert "date" in normalized.columns


def test_validate_price_frame_missing_date():
    df = pd.DataFrame({"close": [100.0]})
    try:
        validate_price_frame(df, "AAPL")
        assert False, "Expected MarketDataError"
    except MarketDataError as exc:
        assert exc.error_code == "MARKET_DATA_MISSING_DATE"


def test_cash_ticker_ignored():
    from market_data.store import MarketDataStore

    store = MarketDataStore.default()
    try:
        store.get_prices("CASH", start="2024-01-01", end="2024-01-02")
        assert False, "Expected MarketDataError for CASH"
    except MarketDataError as exc:
        assert exc.error_code == "MARKET_DATA_SKIP"
