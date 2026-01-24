from analytics.definitions import METRICS_REGISTRY


def test_metric_registry_required_keys() -> None:
    required = {
        "twr",
        "mwr",
        "volatility",
        "max_drawdown",
        "sharpe",
        "sortino",
        "var_daily",
        "cvar_daily",
        "attribution_totals",
        "factor_tilts",
        "allocation_effect",
        "selection_effect",
        "interaction_effect",
        "tracking_error",
        "benchmark_volatility",
        "benchmark_correlation",
        "rolling_drawdown",
    }
    missing = required - set(METRICS_REGISTRY.keys())
    assert not missing


def test_metric_registry_schema() -> None:
    for key, metric in METRICS_REGISTRY.items():
        assert metric.get("title"), f"{key} missing title"
        assert metric.get("formula"), f"{key} missing formula"
        assert metric.get("time_basis"), f"{key} missing time_basis"
        assert metric.get("inputs"), f"{key} missing inputs"
        assert metric.get("domain"), f"{key} missing domain"
        invariants = metric.get("invariants")
        assert isinstance(invariants, list) and invariants, f"{key} missing invariants"
