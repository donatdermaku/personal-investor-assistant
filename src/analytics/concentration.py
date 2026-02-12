from __future__ import annotations

import math

import pandas as pd


def compute_hhi(weights: pd.Series) -> float:
    """
    Compute portfolio concentration using Herfindahl-Hirschman Index (HHI).

    Accepts raw or normalized weights. The function normalizes internally.
    Returns NaN when no positive finite weights are available.
    """
    if weights is None or weights.empty:
        return float("nan")

    cleaned = pd.to_numeric(weights, errors="coerce")
    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA).dropna()
    cleaned = cleaned[cleaned > 0]
    if cleaned.empty:
        return float("nan")

    total = float(cleaned.sum())
    if total <= 0:
        return float("nan")

    normalized = cleaned / total
    hhi = float((normalized**2).sum())
    return hhi


def classify_hhi(hhi: float) -> str:
    """HHI threshold bands: diversified <0.15, moderate 0.15-0.25, concentrated >0.25."""
    if hhi is None or not math.isfinite(hhi):
        return "unavailable"
    if hhi < 0.15:
        return "diversified"
    if hhi <= 0.25:
        return "moderate"
    return "concentrated"


def build_hhi_summary(weights: pd.Series) -> dict:
    hhi = compute_hhi(weights)
    if not math.isfinite(hhi):
        return {
            "hhi": None,
            "classification": "unavailable",
            "effective_positions": None,
            "top_weight": None,
            "thresholds": {
                "diversified_lt": 0.15,
                "moderate_lte": 0.25,
            },
        }

    cleaned = pd.to_numeric(weights, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    cleaned = cleaned[cleaned > 0]
    normalized = cleaned / float(cleaned.sum())
    top_weight = float(normalized.max()) if not normalized.empty else None
    effective_positions = float(1.0 / hhi) if hhi > 0 else None

    return {
        "hhi": hhi,
        "classification": classify_hhi(hhi),
        "effective_positions": effective_positions,
        "top_weight": top_weight,
        "thresholds": {
            "diversified_lt": 0.15,
            "moderate_lte": 0.25,
        },
    }
