from __future__ import annotations

import pandas as pd

from src.analytics.correlation import compute_correlation_matrix


def test_correlation_unavailable_for_few_assets() -> None:
    returns = pd.DataFrame({"AAPL": [0.01] * 10})
    payload = compute_correlation_matrix(returns, min_obs=5)
    assert payload["status"] == "unavailable"
    assert "CORR_TOO_FEW_ASSETS" in payload["reasons"]


def test_correlation_sufficient() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01] * 100,
            "MSFT": [0.02] * 100,
        }
    )
    payload = compute_correlation_matrix(returns, min_obs=60)
    assert payload["status"] in ("sufficient", "partial")
    assert len(payload["matrix"]) == 2
