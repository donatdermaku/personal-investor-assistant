from __future__ import annotations

DEPENDENCY_KEYS = ("prices", "benchmark", "risk_free", "macro")

KPI_DEPENDENCIES: dict[str, list[str]] = {
    "twr": ["prices"],
    "mwr": ["prices"],
    "portfolio_value": ["prices"],
    "final_value": ["prices"],
    "max_drawdown": ["prices"],
    "total_return": ["prices"],
    "var_95": ["prices"],
    "cvar_95": ["prices"],
    "volatility": ["prices"],
    "sharpe": ["prices", "risk_free"],
    "allocation_effect": ["prices", "benchmark"],
    "selection_effect": ["prices", "benchmark"],
    "interaction_effect": ["prices", "benchmark"],
    "tracking_error": ["prices", "benchmark"],
    "benchmark_correlation": ["prices", "benchmark"],
    "benchmark_volatility": ["prices", "benchmark"],
    "correlation_matrix": ["prices"],
}


def evaluate_metric_status(
    coverage: dict[str, dict[str, object]],
    dependencies: list[str],
) -> tuple[str, list[str]]:
    has_unknown = False
    reasons: list[str] = []
    for dep in dependencies:
        entry = coverage.get(dep, {})
        status = entry.get("status")
        if status is None:
            has_unknown = True
            continue
        if status == "insufficient":
            entry_reasons = entry.get("reason_codes")
            if isinstance(entry_reasons, list) and entry_reasons:
                reasons.extend([str(reason) for reason in entry_reasons])
            elif dep == "prices":
                reasons.append("PRICE_INSUFFICIENT")
            elif dep == "benchmark":
                reasons.append("BENCHMARK_INSUFFICIENT")
            elif dep == "risk_free":
                reasons.append("RF_MISSING")
            elif dep == "macro":
                reasons.append("MACRO_MISSING")
    if reasons:
        return "insufficient", reasons
    if has_unknown:
        return "unknown", []
    return "sufficient", []
