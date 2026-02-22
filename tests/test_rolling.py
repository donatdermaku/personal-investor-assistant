from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.rolling import compute_rolling_metrics
from src.api import server as api_server


def _build_inputs(n: int = 504) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    daily_returns = pd.Series(0.001 + np.random.normal(0, 0.01, n), index=idx)

    perf = pd.DataFrame(
        {
            "date": idx,
            "daily_return": daily_returns.values,
            "drawdown": np.zeros(n, dtype=float),
        }
    )
    rf = pd.DataFrame(
        {
            "date": idx,
            "rf_daily_return": np.zeros(n, dtype=float),
        }
    )
    return perf, rf


def test_sharpe_annualization() -> None:
    perf, rf = _build_inputs(504)
    result = compute_rolling_metrics(perf, window=252, risk_free_series=rf)

    returns = pd.Series(perf["daily_return"].values)
    window = returns.iloc[-252:]
    expected = (window.mean() * 252.0) / (window.std(ddof=1) * np.sqrt(252.0))

    actual = float(pd.to_numeric(result["rolling_sharpe"], errors="coerce").dropna().iloc[-1])
    assert abs(actual - expected) < 1e-9


def test_sharpe_consistent_with_server() -> None:
    perf, rf = _build_inputs(252)

    rolling = compute_rolling_metrics(perf, window=252, risk_free_series=rf)
    rolling_sharpe = float(pd.to_numeric(rolling["rolling_sharpe"], errors="coerce").dropna().iloc[-1])

    server_metrics = api_server._compute_risk_metrics(
        perf.to_dict(orient="records"),
        rf.to_dict(orient="records"),
    )
    server_sharpe = server_metrics["sharpe"]

    assert server_sharpe is not None
    assert abs(rolling_sharpe - float(server_sharpe)) < 1e-9
