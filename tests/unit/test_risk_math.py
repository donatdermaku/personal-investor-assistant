from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import compute_drawdown
from src.compute.factors import _calc_price_metrics
from src.analytics.risk import compare_var_budget
from tests.utils import assert_close


def test_drawdown_series() -> None:
    values = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0], index=pd.date_range("2024-01-01", periods=5))
    drawdown = compute_drawdown(values)
    assert_close(float(drawdown.min()), -0.25, tol=1e-6)


def test_annualized_volatility_from_prices() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["ABC"] * 4,
            "date": pd.date_range("2024-01-01", periods=4),
            "adj_close": [100.0, 110.0, 99.0, 108.9],
        }
    )
    metrics = _calc_price_metrics(prices)
    vol = float(metrics.loc[0, "Volatility30d"])
    returns = prices["adj_close"].pct_change().dropna()
    expected = float(returns.std(ddof=0) * np.sqrt(252))
    assert_close(vol, expected, tol=1e-10)


def test_compare_var_budget_within_budget() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    portfolio = pd.Series([-0.010, -0.020, 0.005, 0.003, -0.004], index=idx)
    benchmark = pd.Series([-0.015, -0.025, 0.004, 0.002, -0.005], index=idx)
    out = compare_var_budget(portfolio, benchmark, alpha=0.05)
    assert out["status"] == "ok"
    assert out["within_budget"] is True
    assert out["utilization_ratio"] is not None


def test_compare_var_budget_unavailable_without_overlap() -> None:
    portfolio = pd.Series([-0.01, -0.02], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    benchmark = pd.Series([-0.03, -0.01], index=pd.date_range("2024-02-01", periods=2, freq="D"))
    out = compare_var_budget(portfolio, benchmark, alpha=0.05)
    assert out["status"] == "unavailable"
    assert out["reason"] == "NO_OVERLAP"
