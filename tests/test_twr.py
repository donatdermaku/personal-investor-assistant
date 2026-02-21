from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import compute_portfolio_from_ledger, compute_twr


def test_zero_balance_total_loss_is_permanent() -> None:
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    nav = pd.Series([100.0, 0.0, 100.0, 200.0], index=idx)
    cf = pd.Series([100.0], index=[idx[2]])

    twr, _daily = compute_twr(nav, cf)

    assert twr is not None
    assert abs(twr - (-1.0)) < 1e-9


def test_constant_nav_zero_cashflows() -> None:
    idx = pd.date_range("2023-01-02", periods=252, freq="B")
    nav = pd.Series([100.0] * len(idx), index=idx)
    cf = pd.Series([0.0] * len(idx), index=idx)

    twr, _ = compute_twr(nav, cf)

    assert twr is not None
    assert abs(twr) < 1e-9


def test_undefined_period_excluded_not_zeroed() -> None:
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    nav = pd.Series([0.0, 0.0, 0.0], index=idx)
    cf = pd.Series([0.0, 0.0, 0.0], index=idx)

    twr, daily = compute_twr(nav, cf)

    assert twr is None
    assert np.isnan(float(daily.iloc[1]))
    assert np.isnan(float(daily.iloc[2]))


def test_weekend_deposit_snapped_correctly() -> None:
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-06",  # Saturday
                "ticker": "CASH",
                "action": "DEPOSIT",
                "quantity": 0.0,
                "price": 1000.0,
                "fees": 0.0,
            }
        ]
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08", "2024-01-09"]),
            "ticker": ["AAA", "AAA"],
            "adj_close": [100.0, 100.0],
        }
    )

    result = compute_portfolio_from_ledger(ledger, prices)

    assert not result.errors
    assert result.twr is not None
    assert abs(result.twr) < 1e-9
