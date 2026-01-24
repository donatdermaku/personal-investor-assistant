from __future__ import annotations

import pandas as pd

from src.analytics.macro import compute_macro_regime_payload


def _series(start: str, periods: int, value: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"date": dates, "value": [value] * periods})


def test_macro_partial_when_missing_vix() -> None:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    cpi = _series("2023-10-01", 90, 250.0)
    fed = _series("2023-10-01", 90, 5.0)
    vix = pd.DataFrame()
    payload = compute_macro_regime_payload(
        dates,
        cpi,
        fed,
        vix,
        cache_status={"CPIAUCSL": "fresh", "DFF": "fresh", "VIXCLS": "error"},
    )
    assert payload.status == "partial"
    assert "VIXCLS" in payload.missing_series
