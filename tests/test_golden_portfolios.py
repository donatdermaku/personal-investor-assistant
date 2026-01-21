from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.portfolio import compute_drawdown, compute_monthly_returns, compute_portfolio_from_ledger
from src.streamlit_export import export_monthly_returns_csv, export_performance_csv, export_summary_json
from tests.utils import assert_close

GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"


def _prices_scenario_a() -> pd.DataFrame:
    month_ends = pd.date_range("2024-01-31", "2025-01-31", freq="ME")
    records = [{"date": "2024-01-01", "ticker": "AAA", "adj_close": 100.0}]
    price = 102.0
    for d in month_ends:
        records.append({"date": d.strftime("%Y-%m-%d"), "ticker": "AAA", "adj_close": round(price, 6)})
        price *= 1.02
    return pd.DataFrame(records)


def _prices_scenario_b() -> pd.DataFrame:
    month_ends = pd.date_range("2024-02-29", "2025-02-28", freq="ME")
    records = [
        {"date": "2024-02-01", "ticker": "BBB", "adj_close": 100.0},
        {"date": "2024-02-15", "ticker": "BBB", "adj_close": 95.0},
    ]
    price = 90.0
    for d in month_ends:
        records.append({"date": d.strftime("%Y-%m-%d"), "ticker": "BBB", "adj_close": round(price, 6)})
        price *= 0.99
    return pd.DataFrame(records)


def _load_expected_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_expected_monthly(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return pd.Series(df["return"].values, index=df["date"])


def _run_scenario(name: str, prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    scenario_dir = GOLDEN_ROOT / name
    ledger = pd.read_csv(scenario_dir / "transactions.csv")
    expected = _load_expected_metrics(scenario_dir / "expected_metrics.json")
    expected_monthly = _load_expected_monthly(scenario_dir / "expected_monthly_returns.csv")

    result = compute_portfolio_from_ledger(ledger, prices)
    assert not result.errors
    assert not result.daily_values.empty
    assert result.daily_values["value"].isna().sum() == 0

    assert_close(result.twr, expected["twr"], tol=1e-6)
    assert_close(result.mwr, expected["mwr"], tol=1e-6)
    assert_close(result.daily_values["value"].iloc[-1], expected["final_value"], tol=1e-6)

    max_dd = compute_drawdown(result.daily_values["value"]).min()
    assert_close(max_dd, expected["max_drawdown"], tol=1e-6)

    monthly = compute_monthly_returns(result.daily_returns)
    assert list(monthly.index) == list(expected_monthly.index)
    for date, expected_value in expected_monthly.items():
        assert_close(monthly.loc[date], expected_value, tol=1e-6)

    return result.daily_values["value"], result.daily_returns


def test_golden_scenario_a() -> None:
    _run_scenario("scenario_A", _prices_scenario_a())


def test_golden_scenario_b() -> None:
    _run_scenario("scenario_B", _prices_scenario_b())


def test_exports_match_computed(tmp_path: Path) -> None:
    scenario_dir = GOLDEN_ROOT / "scenario_A"
    ledger = pd.read_csv(scenario_dir / "transactions.csv")
    result = compute_portfolio_from_ledger(ledger, _prices_scenario_a())

    export_summary_json(tmp_path / "summary.json", result)
    export_performance_csv(tmp_path / "performance.csv", result)
    export_monthly_returns_csv(tmp_path / "monthly_returns.csv", result)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert_close(summary["twr"], result.twr, tol=1e-6)
    assert_close(summary["mwr"], result.mwr, tol=1e-6)
    assert_close(summary["final_value"], float(result.daily_values["value"].iloc[-1]), tol=1e-6)
    assert_close(summary["max_drawdown"], compute_drawdown(result.daily_values["value"]).min(), tol=1e-6)

    perf = pd.read_csv(tmp_path / "performance.csv")
    perf["date"] = pd.to_datetime(perf["date"])
    expected_perf = result.daily_values.copy()
    expected_perf["daily_return"] = result.daily_returns.reindex(expected_perf.index).fillna(0.0)
    expected_perf["drawdown"] = compute_drawdown(expected_perf["value"])
    expected_perf = expected_perf.reset_index()

    assert list(perf.columns) == ["date", "value", "cash", "daily_return", "drawdown"]
    assert list(perf["date"]) == list(expected_perf["date"])
    for col in ["value", "cash", "daily_return", "drawdown"]:
        for actual, expected in zip(perf[col].values, expected_perf[col].values, strict=True):
            assert_close(float(actual), float(expected), tol=1e-6)

    monthly = pd.read_csv(tmp_path / "monthly_returns.csv")
    monthly["date"] = pd.to_datetime(monthly["date"])
    expected_monthly = compute_monthly_returns(result.daily_returns).reset_index()
    expected_monthly.columns = ["date", "return"]

    assert list(monthly["date"]) == list(expected_monthly["date"])
    for actual, expected in zip(monthly["return"].values, expected_monthly["return"].values, strict=True):
        assert_close(float(actual), float(expected), tol=1e-6)
