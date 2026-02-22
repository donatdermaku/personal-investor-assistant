from __future__ import annotations

from dataclasses import dataclass
import logging

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

logger = logging.getLogger(__name__)


def _select_factor_column(scores: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in scores.columns:
            return column
    return None


def _normalize_weights(weights: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cleaned = cleaned[cleaned > 0]
    return cleaned


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

    total_weight = float(weights.sum())
    if total_weight <= 0:
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
    coverage_map: dict[str, float] = {}
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
        covered_weight = float(aligned_weights.sum())
        if covered_weight <= 0:
            continue
        score_coverage_pct = covered_weight / total_weight if total_weight > 0 else 0.0
        coverage_map[factor_name] = score_coverage_pct
        portfolio_mean = float((portfolio_factor * aligned_weights).sum() / total_weight)
        tilt_value = float(portfolio_mean - universe_mean)

        details_rows.append(
            {
                "factor": factor_name,
                "source_column": column,
                "portfolio_mean": portfolio_mean,
                "universe_mean": universe_mean,
                "tilt": tilt_value,
                "score_coverage_pct": score_coverage_pct,
            }
        )

    if not details_rows:
        return FactorTiltOutput(
            summary={"status": "unavailable", "reason": "MISSING_FACTOR_COLUMNS"},
            details=pd.DataFrame(),
        )

    details = pd.DataFrame(details_rows).sort_values("factor").reset_index(drop=True)
    overall_coverage = min(coverage_map.values()) if coverage_map else 0.0
    if overall_coverage < 0.5:
        logger.warning("Factor tilt estimate has low score coverage: %.4f", overall_coverage)
    summary = {
        "status": "ok",
        "portfolio_tickers_used": len(common),
        "factor_count": int(details.shape[0]),
        "score_coverage_pct": overall_coverage,
        "score_coverage_by_factor": coverage_map,
        "tilts": {
            row["factor"]: float(row["tilt"])
            for _, row in details.iterrows()
        },
    }
    return FactorTiltOutput(summary=summary, details=details)
