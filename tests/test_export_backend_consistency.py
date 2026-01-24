from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.api import server as api_server
from src.portfolio import compute_drawdown, compute_monthly_returns, compute_portfolio_from_ledger
from src.streamlit_export import export_monthly_returns_csv, export_performance_csv, export_summary_json
from tests.utils import assert_close


def _prices_growth(start: str, periods: int, ticker: str, price: float, growth: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="ME")
    values = []
    current = price
    for _ in dates:
        values.append(round(current, 6))
        current *= 1 + growth
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": ticker, "adj_close": values})


def _assert_optional_number(actual: float | None, expected: float | None, tol: float = 1e-6) -> None:
    if expected is None:
        assert actual is None
        return
    assert_close(float(actual), float(expected), tol=tol)


def test_backend_export_consistency(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "ticker": "CASH",
                "action": "DEPOSIT",
                "quantity": 0,
                "price": 10000.0,
                "fees": 0.0,
            },
            {
                "date": "2024-01-31",
                "ticker": "AAA",
                "action": "BUY",
                "quantity": 100,
                "price": 100.0,
                "fees": 0.0,
            },
        ]
    )
    prices = _prices_growth("2024-01-31", 6, "AAA", 100.0, 0.01)
    result = compute_portfolio_from_ledger(ledger, prices)
    assert not result.errors

    run_id = "test-run"
    export_dir = tmp_path / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    export_summary_json(export_dir / "summary.json", result)
    export_performance_csv(export_dir / "performance.csv", result)
    export_monthly_returns_csv(export_dir / "monthly_returns.csv", result)

    original_exports = api_server.EXPORTS_DIR
    api_server.EXPORTS_DIR = tmp_path
    try:
        summary = api_server._load_summary(run_id)
        performance = api_server._load_performance(run_id)
        monthly_returns = api_server._load_monthly_returns(run_id)
    finally:
        api_server.EXPORTS_DIR = original_exports

    assert_close(summary["twr"], result.twr, tol=1e-6)
    _assert_optional_number(summary["mwr"], result.mwr, tol=1e-6)
    assert_close(summary["final_value"], float(result.daily_values["value"].iloc[-1]), tol=1e-6)
    assert_close(summary["max_drawdown"], compute_drawdown(result.daily_values["value"]).min(), tol=1e-6)

    perf_df = pd.DataFrame(performance)
    perf_df["date"] = pd.to_datetime(perf_df["date"])
    expected_perf = result.daily_values.copy()
    expected_perf["daily_return"] = result.daily_returns.reindex(expected_perf.index).fillna(0.0)
    expected_perf["drawdown"] = compute_drawdown(expected_perf["value"])
    expected_perf = expected_perf.reset_index()

    assert list(perf_df["date"]) == list(expected_perf["date"])
    for col in ["value", "cash", "daily_return", "drawdown"]:
        for actual, expected in zip(perf_df[col].values, expected_perf[col].values, strict=True):
            assert_close(float(actual), float(expected), tol=1e-6)

    monthly_df = pd.DataFrame(monthly_returns)
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    expected_monthly = compute_monthly_returns(result.daily_returns).reset_index()
    expected_monthly.columns = ["date", "return"]

    assert list(monthly_df["date"]) == list(expected_monthly["date"])
    for actual, expected in zip(monthly_df["return"].values, expected_monthly["return"].values, strict=True):
        assert_close(float(actual), float(expected), tol=1e-6)
