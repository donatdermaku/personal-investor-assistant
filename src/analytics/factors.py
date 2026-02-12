from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorTiltOutput:
    summary: dict
    details: pd.DataFrame


_FACTOR_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "value": ["value_pct", "ValueScore"],
    "quality": ["quality_pct", "QualityScore"],
    "momentum": ["momentum_pct", "MomScore"],
    "composite": ["composite_pct", "Composite"],
}


def _select_factor_column(scores: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in scores.columns:
            return column
    return None


def _normalize_weights(weights: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cleaned = cleaned[cleaned > 0]
    if cleaned.empty:
        return pd.Series(dtype=float)
    total = float(cleaned.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return cleaned / total


def compute_factor_tilts(
    scores: pd.DataFrame,
    portfolio_weights: pd.Series,
) -> FactorTiltOutput:
    """
    Compute portfolio factor tilts versus universe baseline.

    Tilt = weighted portfolio factor mean - universe factor mean.
    """
    if scores.empty:
        return FactorTiltOutput(
            summary={"status": "unavailable", "reason": "MISSING_SCORES"},
            details=pd.DataFrame(),
        )

    if "ticker" not in scores.columns:
        return FactorTiltOutput(
            summary={"status": "unavailable", "reason": "MISSING_TICKER_COLUMN"},
            details=pd.DataFrame(),
        )

    weights = _normalize_weights(portfolio_weights)
    if weights.empty:
        return FactorTiltOutput(
            summary={"status": "unavailable", "reason": "MISSING_PORTFOLIO_WEIGHTS"},
            details=pd.DataFrame(),
        )

    scoped = scores.copy()
    scoped["ticker"] = scoped["ticker"].astype(str).str.upper()
    scoped = scoped.drop_duplicates(subset=["ticker"], keep="last")
    scoped = scoped.set_index("ticker")

    common = sorted(set(scoped.index) & set(weights.index))
    if not common:
        return FactorTiltOutput(
            summary={"status": "unavailable", "reason": "NO_PORTFOLIO_SCORE_OVERLAP"},
            details=pd.DataFrame(),
        )

    details_rows: list[dict] = []
    for factor_name, candidates in _FACTOR_COLUMN_CANDIDATES.items():
        column = _select_factor_column(scoped, candidates)
        if not column:
            continue

        factor_series = pd.to_numeric(scoped[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if factor_series.empty:
            continue

        universe_mean = float(factor_series.mean())
        portfolio_factor = factor_series.reindex(common).dropna()
        if portfolio_factor.empty:
            continue
        aligned_weights = weights.reindex(portfolio_factor.index).fillna(0.0)
        weight_sum = float(aligned_weights.sum())
        if weight_sum <= 0:
            continue
        aligned_weights = aligned_weights / weight_sum
        portfolio_mean = float((portfolio_factor * aligned_weights).sum())
        tilt_value = float(portfolio_mean - universe_mean)

        details_rows.append(
            {
                "factor": factor_name,
                "source_column": column,
                "portfolio_mean": portfolio_mean,
                "universe_mean": universe_mean,
                "tilt": tilt_value,
            }
        )

    if not details_rows:
        return FactorTiltOutput(
            summary={"status": "unavailable", "reason": "MISSING_FACTOR_COLUMNS"},
            details=pd.DataFrame(),
        )

    details = pd.DataFrame(details_rows).sort_values("factor").reset_index(drop=True)
    summary = {
        "status": "ok",
        "portfolio_tickers_used": len(common),
        "factor_count": int(details.shape[0]),
        "tilts": {
            row["factor"]: float(row["tilt"])
            for _, row in details.iterrows()
        },
    }
    return FactorTiltOutput(summary=summary, details=details)
