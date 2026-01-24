from __future__ import annotations

import pandas as pd

from src.analytics.macro import compute_macro_regime_payload


def test_macro_unavailable_when_missing_series() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    payload = compute_macro_regime_payload(pd.DatetimeIndex(dates), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert payload.status == "unavailable"
