from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.definitions import DEFINITIONS_REGISTRY


REGISTERED_METRICS = frozenset(
    {
        "allocation_effect",
        "attribution_30d",
        "benchmark_alignment",
        "benchmark_correlation",
        "benchmark_volatility",
        "component_risk",
        "composite_pct",
        "correlation_matrix",
        "current_drawdown",
        "cvar_daily",
        "drawdown_1y",
        "factor_tilts",
        "hhi_concentration",
        "interaction_effect",
        "latest_daily_return",
        "max_drawdown",
        "momentum_pct",
        "mwr",
        "piotroski_f",
        "price_spot",
        "quality_pct",
        "return_7d",
        "rolling_drawdown",
        "rolling_volatility",
        "rsi_14",
        "selection_effect",
        "sharpe_1y",
        "sharpe_rolling",
        "sma_20",
        "sma_50",
        "tracking_error",
        "twr",
        "value_pct",
        "var_daily",
        "var_budget_comparison",
        "volatility_30d",
    }
)

REGISTERED_RUN_METRIC_PAYLOAD_KEYS = frozenset(
    {
        "attribution_summary",
        "attribution_timeseries",
        "benchmark_comparison",
        "benchmark_timeseries",
        "concentration",
        "correlation_matrix",
        "diagnostics",
        "factor_tilts",
        "macro",
        "macro_regimes",
        "risk",
        "risk_contribution",
        "rolling_metrics",
    }
)

REGISTERED_METRIC_ARTIFACT_ALIASES = frozenset(
    {
        "attribution-summary",
        "attribution-timeseries",
        "benchmark-comparison",
        "benchmark-timeseries",
        "concentration-summary",
        "correlation-matrix",
        "diagnostics",
        "factor-tilts",
        "macro-context",
        "macro-regime-summary",
        "macro-regimes",
        "risk-contribution",
        "risk-contribution-json",
        "rolling-metrics",
    }
)


def assert_metrics_registered(metric_keys: Iterable[str]) -> None:
    """Raise ValueError if a metric is not explicitly registered."""
    unknown = sorted({key for key in metric_keys if key not in REGISTERED_METRICS})
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unregistered metric(s): {joined}")


def get_exposed_definitions() -> dict[str, dict[str, Any]]:
    """Expose only definitions that are explicitly registered."""
    missing = sorted(metric for metric in REGISTERED_METRICS if metric not in DEFINITIONS_REGISTRY)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Registered metric(s) missing from DEFINITIONS_REGISTRY: {joined}")
    return {metric: DEFINITIONS_REGISTRY[metric] for metric in sorted(REGISTERED_METRICS)}


def filter_registered_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keys that are not registered metrics."""
    return {key: value for key, value in metrics.items() if key in REGISTERED_METRICS}


def assert_run_metric_payload_keys_registered(metric_keys: Iterable[str]) -> None:
    """Raise ValueError when a run payload contains an unregistered metric section key."""
    unknown = sorted({key for key in metric_keys if key not in REGISTERED_RUN_METRIC_PAYLOAD_KEYS})
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unregistered run metric payload key(s): {joined}")


def assert_metric_artifact_aliases_registered(aliases: Iterable[str]) -> None:
    """Raise ValueError when metric artifact aliases are not explicitly registered."""
    unknown = sorted({alias for alias in aliases if alias not in REGISTERED_METRIC_ARTIFACT_ALIASES})
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unregistered metric artifact alias(es): {joined}")
