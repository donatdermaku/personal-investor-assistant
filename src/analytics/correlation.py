from __future__ import annotations

import numpy as np
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


def compute_correlation_matrix_from_cov(
    cov: np.ndarray | None,
    tickers: list[str],
    n_obs: int,
    *,
    min_obs: int = 60,
    excluded: list[dict] | None = None,
) -> dict:
    if cov is None or not tickers:
        return {
            "status": "unavailable",
            "n_obs": int(n_obs),
            "assets_included": [],
            "assets_excluded": excluded or [],
            "matrix": {},
            "reasons": ["CORR_NO_RETURNS"],
        }
    if len(tickers) < 2:
        return {
            "status": "unavailable",
            "n_obs": int(n_obs),
            "assets_included": tickers,
            "assets_excluded": excluded or [],
            "matrix": {},
            "reasons": ["CORR_TOO_FEW_ASSETS"],
        }
    if n_obs < min_obs:
        return {
            "status": "unavailable",
            "n_obs": int(n_obs),
            "assets_included": tickers,
            "assets_excluded": excluded or [],
            "matrix": {},
            "reasons": ["CORR_INSUFFICIENT_HISTORY"],
        }

    cov = np.asarray(cov, dtype=np.float64)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)
    np.fill_diagonal(corr, 1.0)
    corr = np.round(corr, 4)

    matrix = {
        tickers[i]: {tickers[j]: float(corr[i, j]) for j in range(len(tickers))}
        for i in range(len(tickers))
    }
    reasons: list[str] = []
    status = "sufficient"
    if excluded:
        status = "partial"
        reasons.append("CORR_PARTIAL_ASSETS")
    return {
        "status": status,
        "n_obs": int(n_obs),
        "assets_included": tickers,
        "assets_excluded": excluded or [],
        "matrix": matrix,
        "reasons": reasons,
    }
