from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.concentration import build_hhi_summary
from src.analytics.factors import compute_factor_tilts
from src.analytics.risk import compare_var_budget
from tests.utils import assert_close

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "value_engine"


def _load_expected() -> dict:
    return json.loads((GOLDEN_DIR / "expected_metrics.json").read_text(encoding="utf-8"))


def test_golden_hhi_factor_tilts_and_var_budget() -> None:
    expected = _load_expected()

    weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
    hhi = build_hhi_summary(weights)
    assert_close(float(hhi["hhi"]), expected["hhi_concentration"]["hhi"], tol=1e-12)
    assert hhi["classification"] == expected["hhi_concentration"]["classification"]
    assert_close(float(hhi["effective_positions"]), expected["hhi_concentration"]["effective_positions"], tol=1e-12)
    assert_close(float(hhi["top_weight"]), expected["hhi_concentration"]["top_weight"], tol=1e-12)

    scores = pd.DataFrame(
        [
            {"ticker": "AAA", "value_pct": 90.0, "quality_pct": 70.0, "momentum_pct": 60.0, "composite_pct": 80.0},
            {"ticker": "BBB", "value_pct": 50.0, "quality_pct": 40.0, "momentum_pct": 30.0, "composite_pct": 45.0},
            {"ticker": "CCC", "value_pct": 20.0, "quality_pct": 25.0, "momentum_pct": 40.0, "composite_pct": 30.0},
        ]
    )
    factor_output = compute_factor_tilts(scores, weights)
    assert factor_output.summary.get("status") == "ok"
    tilts = factor_output.summary.get("tilts", {})
    for factor_key, expected_value in expected["factor_tilts"].items():
        assert factor_key in tilts
        assert_close(float(tilts[factor_key]), expected_value, tol=1e-12)

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    portfolio = pd.Series([-0.010, -0.020, 0.005, 0.003, -0.004], index=idx)
    benchmark = pd.Series([-0.015, -0.025, 0.004, 0.002, -0.005], index=idx)
    var_budget = compare_var_budget(portfolio, benchmark, alpha=0.05)
    assert var_budget.get("status") == "ok"
    assert_close(float(var_budget["portfolio_var"]), expected["var_budget"]["portfolio_var"], tol=1e-12)
    assert_close(float(var_budget["benchmark_var"]), expected["var_budget"]["benchmark_var"], tol=1e-12)
    assert_close(float(var_budget["utilization_ratio"]), expected["var_budget"]["utilization_ratio"], tol=1e-12)
    assert var_budget["within_budget"] == expected["var_budget"]["within_budget"]
