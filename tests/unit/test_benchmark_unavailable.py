from __future__ import annotations

import pandas as pd

from src.analytics.comparative import compute_benchmark_comparison


def test_benchmark_unavailable_when_missing() -> None:
    portfolio_returns = pd.Series(dtype=float)
    portfolio_values = pd.DataFrame()
    benchmark_prices = pd.DataFrame()
    output = compute_benchmark_comparison(portfolio_returns, portfolio_values, benchmark_prices)
    assert output.summary.get("status") == "unavailable"
    assert "MISSING_BENCHMARK" in output.summary.get("reasons", [])


def test_benchmark_unavailable_when_no_overlap() -> None:
    portfolio_returns = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    portfolio_values = pd.DataFrame({"value": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    benchmark_prices = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "adj_close": [100.0, 101.0], "ticker": ["SPY", "SPY"]}
    )
    output = compute_benchmark_comparison(portfolio_returns, portfolio_values, benchmark_prices)
    assert output.summary.get("status") == "unavailable"
    assert "NO_OVERLAP" in output.summary.get("reasons", [])
