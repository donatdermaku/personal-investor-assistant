from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.rolling import compute_rolling_metrics
from src.api import server as api_server
from src.portfolio import compute_drawdown, compute_portfolio_from_ledger


def _build_perf(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    daily = pd.Series(0.001 + np.linspace(-0.002, 0.002, n), index=idx)
    value = 100.0 * (1 + daily).cumprod()
    return pd.DataFrame(
        {
            "date": idx,
            "daily_return": daily.values,
            "drawdown": compute_drawdown(value).values,
        }
    )


def test_partial_rf_produces_nan_sharpe() -> None:
    perf = _build_perf(30)
    idx = pd.to_datetime(perf["date"])
    rf = pd.DataFrame(
        {
            "date": idx,
            "rf_daily_return": [np.nan] * 15 + [0.0001] * 15,
        }
    )
    out = compute_rolling_metrics(perf, window=10, risk_free_series=rf)
    sharpe = pd.to_numeric(out["rolling_sharpe"], errors="coerce")
    assert sharpe.isna().any()


def test_full_rf_coverage_matches_phase_0() -> None:
    fixtures = Path("tests/fixtures")
    golden = json.loads((fixtures / "golden_metrics_phase_0.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(fixtures / "baseline_ledger.csv")
    prices = pd.read_parquet(fixtures / "baseline_prices.parquet")

    result = compute_portfolio_from_ledger(ledger, prices)
    perf = result.daily_values.copy()
    perf["daily_return"] = result.daily_returns.reindex(perf.index).fillna(0.0)
    perf["drawdown"] = compute_drawdown(perf["value"]) 
    perf = perf.reset_index().rename(columns={"index": "date"})

    rf = pd.DataFrame(
        {
            "date": pd.to_datetime(perf["date"]),
            "rf_daily_return": np.zeros(len(perf), dtype=float),
        }
    )
    out = compute_rolling_metrics(perf, window=63, risk_free_series=rf)
    actual = float(pd.to_numeric(out["rolling_sharpe"], errors="coerce").dropna().iloc[-1])
    assert abs(actual - float(golden["sharpe_rolling_last"])) < 1e-9


def test_rf_entirely_missing_all_nan() -> None:
    perf = _build_perf(30)
    rf = pd.DataFrame(
        {
            "date": pd.to_datetime(perf["date"]),
            "rf_daily_return": [np.nan] * len(perf),
        }
    )
    out = compute_rolling_metrics(perf, window=10, risk_free_series=rf)
    assert pd.to_numeric(out["rolling_sharpe"], errors="coerce").isna().all()

    metrics = api_server._compute_risk_metrics(perf.to_dict(orient="records"), rf.to_dict(orient="records"))
    assert metrics["sharpe"] is None


def test_negative_rf_no_sign_error() -> None:
    perf = _build_perf(40)
    rf = pd.DataFrame(
        {
            "date": pd.to_datetime(perf["date"]),
            "rf_daily_return": np.full(len(perf), -0.0002, dtype=float),
        }
    )
    metrics = api_server._compute_risk_metrics(perf.to_dict(orient="records"), rf.to_dict(orient="records"))
    assert "sharpe" in metrics
    assert metrics["rf_coverage_pct"] is not None
