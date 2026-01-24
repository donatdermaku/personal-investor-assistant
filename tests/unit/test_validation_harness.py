from __future__ import annotations

import pandas as pd

from analytics.validation import (
    validate_benchmark_alignment,
    validate_prices_vs_trades,
    validate_risk_metrics,
)
from src.api import server as api_server
from src.portfolio import compute_monthly_returns


def test_validate_prices_vs_trades_missing_trade_date() -> None:
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-02-15",
                "ticker": "AAA",
                "action": "BUY",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-02-29", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2024-03-31", "ticker": "AAA", "adj_close": 101.0},
        ]
    )
    result = validate_prices_vs_trades(prices, ledger)
    assert not result.ok()


def test_validate_prices_vs_trades_missing_history() -> None:
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "ticker": "AAA",
                "action": "BUY",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
            },
            {
                "date": "2024-04-30",
                "ticker": "AAA",
                "action": "SELL",
                "quantity": 10,
                "price": 110.0,
                "fees": 0.0,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-02-29", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2024-03-31", "ticker": "AAA", "adj_close": 101.0},
        ]
    )
    result = validate_prices_vs_trades(prices, ledger)
    assert not result.ok()


def test_validate_benchmark_alignment() -> None:
    benchmark = pd.DataFrame(
        [
            {"date": "2024-01-31", "adj_close": 100.0},
            {"date": "2024-02-29", "adj_close": 110.0},
        ]
    )
    portfolio_values = pd.Series([1000.0, 1100.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    result = validate_benchmark_alignment(benchmark, portfolio_values)
    assert result.ok(), result.errors


def test_validate_risk_metrics_monotonicity() -> None:
    perf = [
        {"daily_return": -0.05},
        {"daily_return": -0.02},
        {"daily_return": 0.01},
        {"daily_return": 0.02},
    ]
    metrics = api_server._compute_risk_metrics(perf)
    result = validate_risk_metrics(metrics)
    assert result.ok(), result.errors


def test_monthly_returns_stability() -> None:
    daily = pd.Series(
        [0.0, 0.0, 0.0, 0.0],
        index=pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-28", "2024-03-01"]),
    )
    monthly = compute_monthly_returns(daily)
    assert monthly.notna().all()
