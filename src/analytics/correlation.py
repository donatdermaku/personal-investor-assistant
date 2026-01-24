from __future__ import annotations

import pandas as pd


def compute_correlation_matrix(
    returns: pd.DataFrame,
    *,
    min_obs: int = 60,
    coverage_threshold: float = 0.9,
) -> dict:
    if returns is None or returns.empty:
        return {
            "status": "unavailable",
            "n_obs": 0,
            "assets_included": [],
            "assets_excluded": [],
            "matrix": {},
            "reasons": ["CORR_NO_RETURNS"],
        }
    returns = returns.copy()
    if "CASH" in returns.columns:
        returns = returns.drop(columns=["CASH"])
    if returns.empty:
        return {
            "status": "unavailable",
            "n_obs": 0,
            "assets_included": [],
            "assets_excluded": [],
            "matrix": {},
            "reasons": ["CORR_NO_ASSETS"],
        }

    counts = returns.notna().sum()
    total_obs = len(returns.index)
    included = counts[counts >= min_obs].index.tolist()
    excluded = [
        {"ticker": ticker, "reason": "INSUFFICIENT_OBS"}
        for ticker in returns.columns
        if ticker not in included
    ]
    if len(included) < 2:
        return {
            "status": "unavailable",
            "n_obs": int(total_obs),
            "assets_included": included,
            "assets_excluded": excluded,
            "matrix": {},
            "reasons": ["CORR_TOO_FEW_ASSETS"],
        }

    filtered = returns[included].dropna(how="all")
    if len(filtered.index) < min_obs:
        return {
            "status": "unavailable",
            "n_obs": int(len(filtered.index)),
            "assets_included": included,
            "assets_excluded": excluded,
            "matrix": {},
            "reasons": ["CORR_INSUFFICIENT_HISTORY"],
        }

    corr = filtered.corr().round(4)
    matrix = {
        row: {col: float(corr.loc[row, col]) for col in corr.columns}
        for row in corr.index
    }
    status = "sufficient" if len(excluded) == 0 else "partial"
    reasons = []
    if excluded:
        reasons.append("CORR_PARTIAL_ASSETS")
    return {
        "status": status,
        "n_obs": int(len(filtered.index)),
        "assets_included": included,
        "assets_excluded": excluded,
        "matrix": matrix,
        "reasons": reasons,
    }
