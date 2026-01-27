from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.correlation import compute_correlation_matrix_from_cov
from src.analytics.streaming import OnlineCovariance, iter_price_state


def test_returns_dtype_float32() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "AAA"],
            "adj_close": [100.0, 101.0],
        }
    )
    calendar = pd.date_range("2024-01-01", periods=2, freq="D")
    for _, _, returns in iter_price_state(prices, ["AAA"], calendar):
        assert returns.dtype == np.float32


def test_online_correlation_matrix() -> None:
    returns = np.array(
        [
            [0.01, 0.02, -0.01],
            [0.02, 0.04, 0.00],
            [0.03, 0.06, 0.01],
        ],
        dtype=np.float64,
    )
    online = OnlineCovariance(3)
    for row in returns:
        online.update(row)
    cov = online.covariance()
    payload = compute_correlation_matrix_from_cov(
        cov,
        ["AAA", "BBB", "CCC"],
        n_obs=len(returns),
        min_obs=1,
    )
    matrix = payload["matrix"]
    assert payload["status"] == "sufficient"
    assert abs(matrix["AAA"]["BBB"] - 1.0) < 1e-6
    assert abs(matrix["AAA"]["AAA"] - 1.0) < 1e-6
    assert abs(matrix["AAA"]["CCC"] - matrix["CCC"]["AAA"]) < 1e-6
