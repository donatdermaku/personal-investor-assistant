from __future__ import annotations

import pandas as pd

from src.analytics.comparative import compute_benchmark_comparison


def test_benchmark_alignment_uses_intersection() -> None:
    portfolio_returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    portfolio_values = pd.DataFrame(
        {"value": [100.0, 102.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    benchmark_prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "adj_close": [100.0, 101.0],
            "ticker": ["SPY", "SPY"],
        }
    )
    output = compute_benchmark_comparison(portfolio_returns, portfolio_values, benchmark_prices)
    dates = output.timeseries["date"].tolist()
    assert dates == ["2024-01-02"]
