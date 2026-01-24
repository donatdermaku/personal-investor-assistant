from __future__ import annotations

import pandas as pd

from src.risk_free import compute_risk_free_series


def test_risk_free_series_alignment() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    result = compute_risk_free_series(dates)
    if result.series.empty:
        # Allow empty when cache is unavailable in tests.
        assert result.status == "unavailable"
        return
    assert list(result.series["date"]) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert "rf_daily_return" in result.series.columns
