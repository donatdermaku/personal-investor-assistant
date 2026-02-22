from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.factors import compute_factor_tilts


def test_full_coverage_tilt_unchanged() -> None:
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
    # Full coverage should match prior renormalized behavior.
    expected_value_tilt = (0.6 * 90.0 + 0.4 * 50.0) - ((90.0 + 50.0 + 20.0) / 3.0)
    assert out.summary["tilts"]["value"] == pytest.approx(expected_value_tilt)


def test_partial_coverage_no_renormalization() -> None:
    scores = pd.DataFrame(
        [
            {"ticker": "AAA", "value_pct": 90.0, "quality_pct": 70.0, "momentum_pct": 60.0, "composite_pct": 80.0},
            {"ticker": "CCC", "value_pct": 20.0, "quality_pct": 25.0, "momentum_pct": 40.0, "composite_pct": 30.0},
        ]
    )
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    out = compute_factor_tilts(scores, weights)
    # Covered weight is 0.5; no renormalization means portfolio mean is 0.5 * score.
    expected_portfolio_mean = 0.5 * 90.0
    expected_universe_mean = (90.0 + 20.0) / 2.0
    assert out.summary["tilts"]["value"] == pytest.approx(expected_portfolio_mean - expected_universe_mean)


def test_coverage_pct_reported() -> None:
    scores = pd.DataFrame(
        [
            {"ticker": "AAA", "value_pct": 90.0, "quality_pct": 70.0, "momentum_pct": 60.0, "composite_pct": 80.0},
            {"ticker": "CCC", "value_pct": 20.0, "quality_pct": 25.0, "momentum_pct": 40.0, "composite_pct": 30.0},
        ]
    )
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    out = compute_factor_tilts(scores, weights)
    assert out.summary["score_coverage_pct"] == pytest.approx(0.5)
    assert "score_coverage_by_factor" in out.summary
    assert out.summary["score_coverage_by_factor"]["value"] == pytest.approx(0.5)


def test_low_coverage_warning_logged(caplog) -> None:
    scores = pd.DataFrame([{"ticker": "AAA", "value_pct": 90.0, "quality_pct": 70.0, "momentum_pct": 60.0, "composite_pct": 80.0}])
    weights = pd.Series({"AAA": 0.4, "BBB": 0.6})
    with caplog.at_level("WARNING"):
        out = compute_factor_tilts(scores, weights)
    assert out.summary["status"] == "ok"
    assert out.summary["score_coverage_pct"] < 0.5
    assert any("low score coverage" in rec.message for rec in caplog.records)


def test_compute_factor_tilts_unavailable_without_weights() -> None:
    scores = pd.DataFrame([{"ticker": "AAA", "value_pct": 80.0}])
    out = compute_factor_tilts(scores, pd.Series(dtype=float))
    assert out.summary["status"] == "unavailable"
    assert out.summary["reason"] == "MISSING_PORTFOLIO_WEIGHTS"
