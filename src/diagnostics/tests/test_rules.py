from __future__ import annotations

from src.diagnostics import rules


def test_concentration_risk_triggers() -> None:
    contributions = [
        {"ticker": "AAA", "volatility_pct": 0.4},
        {"ticker": "BBB", "volatility_pct": 0.2},
        {"ticker": "CCC", "volatility_pct": 0.1},
    ]
    signal = rules.concentration_risk(contributions, as_of="2024-01-31", coverage_summary={"metric_status": {"volatility": "sufficient"}})
    assert signal is not None
    assert signal.key == "CONCENTRATION_RISK"


def test_single_asset_dominance_triggers() -> None:
    signal = rules.single_asset_dominance(
        {"AAPL": 0.45, "MSFT": 0.2},
        as_of="2024-01-31",
        coverage_summary={"metric_status": {"portfolio_value": "sufficient"}},
    )
    assert signal is not None
    assert signal.severity == "high"


def test_drawdown_sensitivity_triggers() -> None:
    signal = rules.drawdown_sensitivity(
        portfolio_drawdown=-0.25,
        benchmark_drawdown=-0.10,
        as_of="2024-01-31",
        coverage_summary={"metric_status": {"max_drawdown": "sufficient", "benchmark_correlation": "sufficient"}},
    )
    assert signal is not None
    assert signal.key == "DRAWNDOWN_SENSITIVITY"


def test_return_driver_mismatch_triggers() -> None:
    signal = rules.return_driver_mismatch(
        allocation=0.12,
        selection=-0.04,
        total_return=0.13,
        as_of="2024-01-31",
        coverage_summary={"metric_status": {"allocation_effect": "sufficient"}},
    )
    assert signal is not None
    assert signal.key == "RETURN_DRIVER_MISMATCH"


def test_risk_free_dependency_warning_triggers() -> None:
    signal = rules.risk_free_dependency_warning(
        {"metric_status": {"sharpe": "insufficient", "twr": "sufficient"}, "metric_reasons": {"sharpe": ["RF_MISSING"]}},
        as_of="2024-01-31",
    )
    assert signal is not None
    assert signal.category == "data"


def test_benchmark_mismatch_triggers() -> None:
    signal = rules.benchmark_mismatch(
        tracking_error=0.01,
        correlation=0.4,
        as_of="2024-01-31",
        coverage_summary={"metric_status": {"tracking_error": "sufficient", "benchmark_correlation": "sufficient"}},
    )
    assert signal is not None
    assert signal.key == "BENCHMARK_MISMATCH"


def test_short_history_warning_triggers() -> None:
    signal = rules.short_history_warning(
        {
            "policy": {"min_history_days": 252},
            "per_ticker": {"AAPL": {"history_days": 120}},
        },
        as_of="2024-01-31",
    )
    assert signal is not None
    assert signal.key == "SHORT_HISTORY_WARNING"
