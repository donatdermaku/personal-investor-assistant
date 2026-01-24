from __future__ import annotations

import pandas as pd

from src.analytics.comparative import compute_benchmark_comparison


def test_benchmark_unavailable_when_missing() -> None:
    portfolio_returns = pd.Series(dtype=float)
    portfolio_values = pd.DataFrame()
    benchmark_prices = pd.DataFrame()
    output = compute_benchmark_comparison(portfolio_returns, portfolio_values, benchmark_prices)
    assert output.summary.get("status") == "unavailable"
