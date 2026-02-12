from __future__ import annotations

import pandas as pd

from src.analytics.concentration import build_hhi_summary, classify_hhi, compute_hhi


def test_compute_hhi_normalizes_raw_weights() -> None:
    weights = pd.Series({"A": 50, "B": 30, "C": 20})
    hhi = compute_hhi(weights)
    assert round(hhi, 6) == 0.38


def test_hhi_classification_thresholds() -> None:
    assert classify_hhi(0.10) == "diversified"
    assert classify_hhi(0.20) == "moderate"
    assert classify_hhi(0.30) == "concentrated"


def test_hhi_summary_handles_empty_weights() -> None:
    summary = build_hhi_summary(pd.Series(dtype=float))
    assert summary["hhi"] is None
    assert summary["classification"] == "unavailable"
