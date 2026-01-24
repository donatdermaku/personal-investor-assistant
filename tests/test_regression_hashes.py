from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from analytics.validation import validate_regression_hash
from src.portfolio import compute_monthly_returns, compute_portfolio_from_ledger

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "portfolio_multi_asset"


def _prices_growth(start: str, periods: int, ticker: str, price: float, growth: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="ME")
    values = []
    current = price
    for _ in dates:
        values.append(round(current, 6))
        current *= 1 + growth
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": ticker, "adj_close": values})


def _prices_multi(start: str, periods: int) -> pd.DataFrame:
    aaa = _prices_growth(start, periods, "AAA", 100.0, 0.02)
    bbb = _prices_growth(start, periods, "BBB", 50.0, -0.01)
    return pd.concat([aaa, bbb], ignore_index=True)


def test_regression_hashes_portfolio_multi_asset() -> None:
    ledger = pd.read_csv(GOLDEN_DIR / "transactions.csv")
    prices = _prices_multi("2024-01-31", 12)
    result = compute_portfolio_from_ledger(ledger, prices)
    assert not result.errors

    expected = json.loads((GOLDEN_DIR / "expected_hashes.json").read_text())

    equity_series = [
        (d.strftime("%Y-%m-%d"), float(v)) for d, v in result.daily_values["value"].items()
    ]
    monthly = compute_monthly_returns(result.daily_returns)
    monthly_series = [(d.strftime("%Y-%m-%d"), float(v)) for d, v in monthly.items()]
    summary_payload = f"twr:{result.twr}|mwr:{result.mwr}|final:{float(result.daily_values['value'].iloc[-1])}"

    equity_result = validate_regression_hash(equity_series, expected["equity_curve"])
    assert equity_result.ok(), equity_result.errors

    monthly_result = validate_regression_hash(monthly_series, expected["monthly_returns"])
    assert monthly_result.ok(), monthly_result.errors

    summary_hash = hashlib.sha256(summary_payload.encode("utf-8")).hexdigest()
    assert summary_hash == expected["summary"]
