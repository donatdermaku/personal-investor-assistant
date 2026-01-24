from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.attribution import compute_attribution
from src.analytics.comparative import compute_benchmark_comparison
from src.analytics.rolling import compute_rolling_metrics
from src.analytics.risk import compute_risk_contributions
from src.api import server as api_server
from src.portfolio import compute_drawdown, compute_monthly_returns, compute_portfolio_from_ledger
from src.streamlit_export import (
    export_attribution_summary_json,
    export_attribution_timeseries_csv,
    export_benchmark_comparison_json,
    export_benchmark_timeseries_csv,
    export_coverage_summary_json,
    export_macro_regime_flags_csv,
    export_macro_regime_summary_json,
    export_monthly_returns_csv,
    export_performance_csv,
    export_risk_free_series_csv,
    export_risk_contribution_csv,
    export_risk_contribution_json,
    export_rolling_metrics_csv,
    export_summary_json,
)
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

    attribution = compute_attribution(prices, result.holdings_daily, result.daily_values, result.daily_returns)
    export_attribution_summary_json(export_dir / "attribution_summary.json", attribution.summary)
    export_attribution_timeseries_csv(export_dir / "attribution_timeseries.csv", attribution.timeseries)

    returns = (
        prices.pivot_table(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .pct_change()
        .fillna(0.0)
    )
    latest = result.holdings_daily.copy()
    latest_date = pd.to_datetime(latest["date"]).max()
    latest = latest[pd.to_datetime(latest["date"]) == latest_date]
    prices_at_date = prices.copy()
    prices_at_date["date"] = pd.to_datetime(prices_at_date["date"])
    prices_at_date = prices_at_date[prices_at_date["date"] == latest_date]
    weights = latest.set_index("ticker")["quantity"] * prices_at_date.set_index("ticker")["adj_close"]
    weights = weights / float(result.daily_values["value"].iloc[-1])
    risk_output = compute_risk_contributions(returns, weights)
    export_risk_contribution_csv(export_dir / "risk_contribution.csv", risk_output.contributions)
    export_risk_contribution_json(export_dir / "risk_contribution.json", risk_output.summary, risk_output.contributions)

    perf = pd.read_csv(export_dir / "performance.csv")
    rolling = compute_rolling_metrics(perf)
    export_rolling_metrics_csv(export_dir / "rolling_metrics.csv", rolling)

    macro_flags = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "inflation_yoy": 0.02,
                "fed_funds": 5.25,
                "vix": 18.0,
                "rates_change_6m": 0.0,
                "high_inflation": False,
                "rising_rates": False,
                "risk_off": False,
            }
        ]
    )
    export_macro_regime_flags_csv(export_dir / "macro_regime_flags.csv", macro_flags)
    export_coverage_summary_json(
        export_dir / "coverage_summary.json",
        {
            "as_of": "2024-01-31",
            "status": "sufficient",
            "score": 1.0,
            "policy": {
                "min_score_for_kpis": 0.95,
                "min_history_days": 252,
                "max_gap_days": 5,
            },
            "required": {"tickers": ["AAA"], "history_days_needed": 252},
            "per_ticker": {},
            "aggregate": {
                "coverage_ratio": 1.0,
                "min_ticker_score": 1.0,
                "benchmark_score": None,
                "rf_score": None,
            },
            "reason_codes": ["OK"],
        },
    )
    export_macro_regime_summary_json(
        export_dir / "macro_regime_summary.json",
        {"status": "ok", "missing_series": [], "as_of": "2024-01-31"},
    )

    benchmark_prices = _prices_growth("2024-01-31", 6, "SPY", 100.0, 0.005)
    comparison = compute_benchmark_comparison(result.daily_returns, result.daily_values, benchmark_prices)
    export_benchmark_comparison_json(export_dir / "benchmark_comparison.json", comparison.summary)
    export_benchmark_timeseries_csv(export_dir / "benchmark_timeseries.csv", comparison.timeseries)
    risk_free_series = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-02-01"],
            "rate": [0.05, 0.05],
            "rf_daily_return": [0.0002, 0.0002],
        }
    )
    export_risk_free_series_csv(export_dir / "risk_free_series.csv", risk_free_series)
    corporate_actions = pd.DataFrame(
        {
            "date": ["2024-02-01"],
            "ticker": ["AAA"],
            "dividend": [0.5],
            "split_ratio": [1.0],
        }
    )
    corporate_actions.to_csv(export_dir / "corporate_actions_events.csv", index=False)

    original_exports = api_server.EXPORTS_DIR
    api_server.EXPORTS_DIR = tmp_path
    try:
        summary = api_server._load_summary(run_id)
        performance = api_server._load_performance(run_id)
        monthly_returns = api_server._load_monthly_returns(run_id)
        attribution_summary = api_server._load_attribution_summary(run_id)
        attribution_timeseries = api_server._load_attribution_timeseries(run_id)
        risk_payload = api_server._load_risk_contribution(run_id)
        rolling_metrics = api_server._load_rolling_metrics(run_id)
        macro = api_server._load_macro_regimes(run_id)
        macro_summary = api_server._load_macro_summary(run_id)
        coverage_summary = api_server._load_coverage_summary(run_id)
        risk_free = api_server._load_risk_free_series(run_id)
        corporate_actions_loaded = api_server._load_corporate_actions(run_id)
        bench_summary = api_server._load_benchmark_comparison(run_id)
        bench_timeseries = api_server._load_benchmark_timeseries(run_id)
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

    assert_close(attribution_summary["allocation"], attribution.summary["allocation"], tol=1e-6)
    assert_close(attribution_summary["selection"], attribution.summary["selection"], tol=1e-6)
    assert_close(attribution_summary["interaction"], attribution.summary["interaction"], tol=1e-6)
    assert len(attribution_timeseries) == len(attribution.timeseries)
    assert len(risk_payload["contributions"]) == len(risk_output.contributions)
    assert len(rolling_metrics) == len(rolling)
    assert len(macro) == len(macro_flags)
    assert macro_summary.get("status") == "ok"
    assert coverage_summary.get("status") == "sufficient"
    assert len(risk_free) == len(risk_free_series)
    assert len(corporate_actions_loaded) == len(corporate_actions)
    assert_close(bench_summary.get("tracking_error"), comparison.summary.get("tracking_error"), tol=1e-6)
    assert len(bench_timeseries) == len(comparison.timeseries)
