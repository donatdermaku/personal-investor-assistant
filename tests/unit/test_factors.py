from __future__ import annotations

import pandas as pd

from src.analytics.factors import compute_factor_tilts


def test_compute_factor_tilts_ok() -> None:
    scores = pd.DataFrame(
        [
            {"ticker": "AAA", "value_pct": 90.0, "quality_pct": 70.0, "momentum_pct": 60.0, "composite_pct": 80.0},
            {"ticker": "BBB", "value_pct": 50.0, "quality_pct": 40.0, "momentum_pct": 30.0, "composite_pct": 45.0},
            {"ticker": "CCC", "value_pct": 20.0, "quality_pct": 25.0, "momentum_pct": 40.0, "composite_pct": 30.0},
        ]
    )
    weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
    out = compute_factor_tilts(scores, weights)
    assert out.summary["status"] == "ok"
    assert out.summary["factor_count"] >= 3
    assert "value" in out.summary["tilts"]
    assert not out.details.empty


def test_compute_factor_tilts_unavailable_without_weights() -> None:
    scores = pd.DataFrame([{"ticker": "AAA", "value_pct": 80.0}])
    out = compute_factor_tilts(scores, pd.Series(dtype=float))
    assert out.summary["status"] == "unavailable"
    assert out.summary["reason"] == "MISSING_PORTFOLIO_WEIGHTS"
