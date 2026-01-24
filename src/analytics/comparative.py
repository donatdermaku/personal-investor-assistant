from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BenchmarkComparisonOutput:
    summary: dict
    timeseries: pd.DataFrame


def compute_benchmark_comparison(
    portfolio_returns: pd.Series,
    portfolio_values: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> BenchmarkComparisonOutput:
    if portfolio_returns.empty or benchmark_prices.empty:
        return BenchmarkComparisonOutput(
            summary={"status": "unavailable", "reason": "MISSING_BENCHMARK", "reasons": ["MISSING_BENCHMARK"]},
            timeseries=pd.DataFrame(),
        )

    bench = benchmark_prices.copy()
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
    bench = bench.dropna(subset=["date", "adj_close"])
    bench = bench.sort_values("date")
    bench_returns = bench.set_index("date")["adj_close"].pct_change().fillna(0.0)

    portfolio_returns = portfolio_returns.copy()
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index, errors="coerce")
    portfolio_returns = portfolio_returns.dropna()

    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), bench_returns.rename("benchmark")],
        axis=1,
        join="inner",
    )
    if aligned.empty:
        return BenchmarkComparisonOutput(
            summary={"status": "unavailable", "reason": "NO_OVERLAP", "reasons": ["NO_OVERLAP"]},
            timeseries=pd.DataFrame(),
        )

    active = aligned["portfolio"] - aligned["benchmark"]
    tracking_error = float(active.std(ddof=1) * np.sqrt(252)) if active.size > 1 else None

    vol_p = float(aligned["portfolio"].std(ddof=1) * np.sqrt(252)) if aligned["portfolio"].size > 1 else None
    vol_b = float(aligned["benchmark"].std(ddof=1) * np.sqrt(252)) if aligned["benchmark"].size > 1 else None
    corr = float(aligned["portfolio"].corr(aligned["benchmark"])) if aligned["portfolio"].size > 1 else None
    implied_te = None
    if vol_p is not None and vol_b is not None and corr is not None:
        implied_te = float(np.sqrt(max(vol_p**2 + vol_b**2 - 2 * corr * vol_p * vol_b, 0.0)))

    drawdown_p = _compute_drawdown_series(portfolio_values)
    drawdown_b = _compute_drawdown_series_from_returns(bench_returns)
    drawdown = pd.concat([drawdown_p.rename("portfolio_dd"), drawdown_b.rename("benchmark_dd")], axis=1, join="inner")
    rel_drawdown = drawdown["portfolio_dd"] - drawdown["benchmark_dd"] if not drawdown.empty else pd.Series(dtype=float)

    timeseries = pd.DataFrame(
        {
            "date": aligned.index.strftime("%Y-%m-%d"),
            "portfolio_return": aligned["portfolio"].values,
            "benchmark_return": aligned["benchmark"].values,
            "active_return": active.values,
            "relative_drawdown": rel_drawdown.reindex(aligned.index).values if not rel_drawdown.empty else np.nan,
        }
    )

    summary = {
        "tracking_error": tracking_error,
        "portfolio_volatility": vol_p,
        "benchmark_volatility": vol_b,
        "correlation": corr,
        "tracking_error_implied": implied_te,
        "status": "ok",
        "reasons": [],
    }
    return BenchmarkComparisonOutput(summary=summary, timeseries=timeseries)


def _compute_drawdown_series(portfolio_values: pd.DataFrame) -> pd.Series:
    if portfolio_values.empty:
        return pd.Series(dtype=float)
    values = portfolio_values.copy()
    if "value" in values.columns:
        series = values["value"]
    else:
        series = values.iloc[:, 0]
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.dropna()
    peak = series.cummax()
    return series / peak - 1.0


def _compute_drawdown_series_from_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index, errors="coerce")
    idx = returns.dropna()
    if idx.empty:
        return pd.Series(dtype=float)
    levels = (1 + idx).cumprod()
    peak = levels.cummax()
    return levels / peak - 1.0
