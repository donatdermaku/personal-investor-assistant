from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analytics.validation import validate_portfolio_result
from src.portfolio import compute_drawdown, compute_monthly_returns, compute_portfolio_from_ledger
from tests.utils import assert_close

GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"


def _prices_constant(start: str, periods: int, ticker: str, price: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="ME")
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": ticker, "adj_close": price})


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


def _load_expected_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_expected_monthly(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return pd.Series(df["return"].values, index=df["date"])


def _assert_optional_number(actual: float | None, expected: float | None, tol: float = 1e-6) -> None:
    if expected is None:
        assert actual is None or (isinstance(actual, float) and pd.isna(actual))
        return
    assert_close(float(actual), float(expected), tol=tol)


def _run_scenario(name: str, prices: pd.DataFrame) -> None:
    scenario_dir = GOLDEN_ROOT / name
    ledger = pd.read_csv(scenario_dir / "transactions.csv")
    expected = _load_expected_metrics(scenario_dir / "expected_metrics.json")
    expected_monthly = _load_expected_monthly(scenario_dir / "expected_monthly_returns.csv")

    result = compute_portfolio_from_ledger(ledger, prices)
    assert not result.errors
    assert not result.daily_values.empty

    validation = validate_portfolio_result(result)
    assert validation.ok(), validation.errors

    _assert_optional_number(result.twr, expected.get("twr"))
    _assert_optional_number(result.mwr, expected.get("mwr"))
    _assert_optional_number(
        float(result.daily_values["value"].iloc[-1]) if not result.daily_values.empty else None,
        expected.get("final_value"),
    )

    max_dd = compute_drawdown(result.daily_values["value"]).min()
    if isinstance(max_dd, float) and pd.isna(max_dd):
        max_dd = None
    _assert_optional_number(max_dd, expected.get("max_drawdown"))

    monthly = compute_monthly_returns(result.daily_returns)
    assert list(monthly.index) == list(expected_monthly.index)
    for date, expected_value in expected_monthly.items():
        _assert_optional_number(monthly.loc[date], expected_value, tol=1e-6)


def test_golden_portfolio_no_trades() -> None:
    _run_scenario("portfolio_no_trades", _prices_constant("2024-01-31", 6, "AAA", 100.0))


def test_golden_portfolio_cash_only() -> None:
    _run_scenario("portfolio_cash_only", _prices_constant("2024-01-31", 6, "AAA", 100.0))


def test_golden_portfolio_single_trade() -> None:
    _run_scenario("portfolio_single_trade", _prices_growth("2024-01-31", 12, "AAA", 100.0, 0.02))


def test_golden_portfolio_dividends_only() -> None:
    _run_scenario("portfolio_dividends_only", _prices_constant("2024-01-31", 6, "AAA", 100.0))


def test_golden_portfolio_multi_asset() -> None:
    _run_scenario("portfolio_multi_asset", _prices_multi("2024-01-31", 12))
