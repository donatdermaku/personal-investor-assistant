from __future__ import annotations

import pandas as pd

from src.analytics.attribution import compute_attribution
from src.analytics.comparative import compute_benchmark_comparison
from src.analytics.macro import compute_macro_regime_payload
from src.analytics.risk import compute_risk_contributions
from src.analytics.rolling import compute_rolling_metrics
from src.portfolio import compute_portfolio_from_ledger
from tests.utils import assert_close


def _prices_growth(start: str, periods: int, ticker: str, price: float, growth: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="ME")
    values = []
    current = price
    for _ in dates:
        values.append(round(current, 6))
        current *= 1 + growth
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": ticker, "adj_close": values})


def test_attribution_sums_to_daily_return() -> None:
    ledger = pd.DataFrame(
        [
            {"date": "2024-01-31", "ticker": "CASH", "action": "DEPOSIT", "quantity": 0, "price": 10000.0, "fees": 0.0},
            {"date": "2024-01-31", "ticker": "AAA", "action": "BUY", "quantity": 50, "price": 100.0, "fees": 0.0},
            {"date": "2024-01-31", "ticker": "BBB", "action": "BUY", "quantity": 50, "price": 100.0, "fees": 0.0},
        ]
    )
    prices = pd.concat(
        [
            _prices_growth("2024-01-31", 6, "AAA", 100.0, 0.02),
            _prices_growth("2024-01-31", 6, "BBB", 100.0, -0.01),
        ],
        ignore_index=True,
    )
    result = compute_portfolio_from_ledger(ledger, prices)
    attribution = compute_attribution(prices, result.holdings_daily, result.daily_values, result.daily_returns)
    assert not attribution.timeseries.empty

    for _, row in attribution.timeseries.iterrows():
        total = row["allocation"] + row["selection"] + row["interaction"]
        assert_close(float(total), float(row["total_return"]), tol=1e-6)

    summary_total = (
        attribution.summary.get("allocation", 0)
        + attribution.summary.get("selection", 0)
        + attribution.summary.get("interaction", 0)
    )
    assert_close(float(summary_total), float(attribution.summary.get("total_return", 0)), tol=1e-6)


def test_risk_contribution_sums() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.02, 0.015, 0.0],
            "BBB": [0.005, -0.01, 0.01, 0.002],
        },
        index=pd.date_range("2024-01-31", periods=4, freq="D"),
    )
    weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
    output = compute_risk_contributions(returns, weights, cash_weight=0.2)
    assert not output.contributions.empty
    total_vol = output.contributions["volatility_contribution"].sum()
    assert_close(float(total_vol), float(output.summary["portfolio_volatility"]), tol=1e-6)


def test_macro_regime_flags_thresholds() -> None:
    daily_dates = pd.date_range("2023-01-01", periods=400, freq="D")
    monthly_dates = pd.date_range("2023-01-01", periods=14, freq="ME")
    cpi = pd.DataFrame({"date": monthly_dates, "value": [100 + 5 * i for i in range(14)]})
    fed = pd.DataFrame({"date": monthly_dates, "value": [4.0 + 0.1 * i for i in range(14)]})
    vix = pd.DataFrame({"date": monthly_dates, "value": [18.0] * 11 + [25.0] * 3})

    payload = compute_macro_regime_payload(pd.DatetimeIndex(daily_dates), cpi, fed, vix)
    assert payload.status == "sufficient"
    assert not payload.flags.empty
    latest = payload.flags.iloc[-1]
    assert bool(latest["high_inflation"]) is True
    assert bool(latest["rising_rates"]) is True
    assert bool(latest["risk_off"]) is True


def test_benchmark_comparison_outputs() -> None:
    ledger = pd.DataFrame(
        [
            {"date": "2024-01-31", "ticker": "CASH", "action": "DEPOSIT", "quantity": 0, "price": 10000.0, "fees": 0.0},
            {"date": "2024-01-31", "ticker": "AAA", "action": "BUY", "quantity": 100, "price": 100.0, "fees": 0.0},
        ]
    )
    prices = _prices_growth("2024-01-31", 6, "AAA", 100.0, 0.01)
    result = compute_portfolio_from_ledger(ledger, prices)
    benchmark_prices = _prices_growth("2024-01-31", 6, "SPY", 100.0, 0.005)
    comparison = compute_benchmark_comparison(result.daily_returns, result.daily_values, benchmark_prices)
    assert comparison.summary
    assert comparison.timeseries is not None


def test_rolling_metrics_window() -> None:
    performance = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=5, freq="D").strftime("%Y-%m-%d"),
            "daily_return": [0.01, -0.02, 0.015, 0.0, 0.005],
            "drawdown": [0.0, -0.02, -0.01, -0.01, -0.005],
        }
    )
    rolling = compute_rolling_metrics(performance, window=3)
    assert len(rolling) == len(performance)
