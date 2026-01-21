from __future__ import annotations

import numpy as np
import pandas as pd


def factor_tilts(scores: pd.DataFrame, tickers: list[str], benchmark_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    if scores.empty or not tickers:
        return pd.DataFrame()
    subset = scores[scores["ticker"].isin(tickers)].copy()
    if subset.empty:
        return pd.DataFrame()
    metrics = ["value_pct", "quality_pct", "momentum_pct", "composite_pct"]
    port = subset[metrics].mean()
    if benchmark_scores is None or benchmark_scores.empty:
        bench = scores[metrics].mean()
    else:
        bench = benchmark_scores[metrics].mean()
    delta = port - bench
    return pd.DataFrame({"portfolio": port, "benchmark": bench, "tilt": delta}).reset_index().rename(columns={"index": "metric"})


def drawdown_intelligence(portfolio_values: pd.Series) -> dict:
    if portfolio_values.empty:
        return {}
    idx = portfolio_values
    peak = idx.cummax()
    drawdown = idx / peak - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    current_dd = float(drawdown.iloc[-1]) if not drawdown.empty else np.nan
    return {
        "max_drawdown": max_dd,
        "current_drawdown": current_dd,
    }


def component_risk(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    cov = returns.cov()
    w = weights.reindex(returns.columns).fillna(0.0)
    if w.sum() == 0:
        return pd.DataFrame()
    portfolio_var = float(w.T @ cov @ w)
    if portfolio_var == 0:
        return pd.DataFrame()
    contrib = (w * (cov @ w)) / portfolio_var
    return pd.DataFrame({"ticker": contrib.index, "risk_contribution": contrib.values}).sort_values("risk_contribution", ascending=False)
