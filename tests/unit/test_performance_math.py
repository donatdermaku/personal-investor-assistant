from __future__ import annotations

import pandas as pd

from src.portfolio import align_benchmark, compute_irr, compute_monthly_returns, compute_twr
from tests.utils import assert_close


def test_twr_no_cashflows() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    values = pd.Series([100.0, 110.0, 121.0], index=dates)
    twr, daily = compute_twr(values, pd.Series(dtype=float))
    assert_close(twr, 0.21, tol=1e-6)
    assert daily.iloc[-1] == daily.iloc[-1]


def test_twr_with_cashflow() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    values = pd.Series([100.0, 160.0, 176.0], index=dates)
    cashflows = pd.Series([50.0], index=[dates[1]])
    twr, _ = compute_twr(values, cashflows)
    assert_close(twr, 0.21, tol=1e-6)


def test_irr_trivial_case() -> None:
    dates = pd.to_datetime(["2024-01-01", "2025-01-01"])
    cashflows = pd.Series([-100.0, 0.0], index=dates)
    irr = compute_irr(cashflows, 110.0)
    assert irr is not None
    assert_close(irr, 0.10, tol=1e-3)


def test_mwr_differs_from_twr_with_late_deposit() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"])
    values = pd.Series([100.0, 1100.0, 1210.0], index=dates)
    cashflows = pd.Series([-100.0, -900.0, 0.0], index=[dates[0], dates[1], dates[2]])
    twr, _ = compute_twr(values, pd.Series([900.0], index=[dates[1]]))
    irr = compute_irr(cashflows, 1210.0)
    assert irr is not None
    assert irr < twr


def test_monthly_returns() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-01"])
    daily = pd.Series([0.10, 0.20], index=dates)
    monthly = compute_monthly_returns(daily)
    assert_close(monthly.loc[pd.Timestamp("2024-01-31")], 0.10, tol=1e-6)
    assert_close(monthly.loc[pd.Timestamp("2024-02-29")], 0.20, tol=1e-6)


def test_benchmark_alignment() -> None:
    bench = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "adj_close": [50.0, 55.0],
        }
    )
    portfolio_values = pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    aligned = align_benchmark(bench, portfolio_values)
    assert_close(aligned.iloc[0], 100.0, tol=1e-6)
    assert_close(aligned.iloc[1], 110.0, tol=1e-6)
