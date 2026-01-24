from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from src.diagnostics.contracts import DiagnosticSignal, DiagnosticsVersion, as_of_label
from src.diagnostics import rules


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _latest_as_of(summary: dict | None) -> str | None:
    if summary is None:
        return None
    return summary.get("last_date") or summary.get("as_of")


def _portfolio_drawdown(performance: list[dict]) -> float | None:
    if not performance:
        return None
    values = [row.get("drawdown") for row in performance if row.get("drawdown") is not None]
    if not values:
        return None
    return float(min(values))


def _benchmark_drawdown(performance: list[dict], benchmark_timeseries: list[dict]) -> float | None:
    if not performance or not benchmark_timeseries:
        return None
    perf_map = {row.get("date"): row.get("drawdown") for row in performance}
    bench_dd = []
    for row in benchmark_timeseries:
        date = row.get("date")
        rel = row.get("relative_drawdown")
        if date is None or rel is None:
            continue
        portfolio_dd = perf_map.get(date)
        if portfolio_dd is None:
            continue
        bench_dd.append(float(portfolio_dd) - float(rel))
    if not bench_dd:
        return None
    return float(min(bench_dd))


def _weights_from_series(weights: pd.Series) -> dict[str, float]:
    if weights is None or weights.empty:
        return {}
    return {str(ticker): float(weight) for ticker, weight in weights.items() if weight is not None}


def generate_diagnostics(
    *,
    summary: dict | None,
    attribution_summary: dict | None,
    risk_contribution: dict | None,
    benchmark_comparison: dict | None,
    benchmark_timeseries: list[dict],
    performance: list[dict],
    rolling_metrics: list[dict],
    coverage_summary: dict | None,
    weights: pd.Series | None = None,
) -> list[DiagnosticSignal]:
    as_of = as_of_label(_latest_as_of(summary))

    contributions = (risk_contribution or {}).get("contributions", [])
    signals: list[DiagnosticSignal] = []

    signals.append(
        rules.concentration_risk(
            contributions,
            as_of=as_of,
            coverage_summary=coverage_summary,
        )
    )

    signals.append(
        rules.single_asset_dominance(
            _weights_from_series(weights),
            as_of=as_of,
            coverage_summary=coverage_summary,
        )
    )

    portfolio_dd = _portfolio_drawdown(performance)
    benchmark_dd = _benchmark_drawdown(performance, benchmark_timeseries)
    signals.append(
        rules.drawdown_sensitivity(
            portfolio_dd,
            benchmark_dd,
            as_of=as_of,
            coverage_summary=coverage_summary,
        )
    )

    if attribution_summary:
        signals.append(
            rules.return_driver_mismatch(
                attribution_summary.get("allocation"),
                attribution_summary.get("selection"),
                attribution_summary.get("total_return"),
                as_of=as_of,
                coverage_summary=coverage_summary,
            )
        )

    signals.append(
        rules.risk_free_dependency_warning(
            coverage_summary,
            as_of=as_of,
        )
    )

    if benchmark_comparison:
        signals.append(
            rules.benchmark_mismatch(
                benchmark_comparison.get("tracking_error"),
                benchmark_comparison.get("correlation"),
                as_of=as_of,
                coverage_summary=coverage_summary,
            )
        )

    signals.append(
        rules.short_history_warning(
            coverage_summary,
            as_of=as_of,
        )
    )

    return _sort_signals(rules.normalize_signals(signals))


def _sort_signals(signals: Iterable[DiagnosticSignal]) -> list[DiagnosticSignal]:
    return sorted(
        signals,
        key=lambda s: (SEVERITY_ORDER.get(s.severity, 3), s.key),
    )


def diagnostics_payload(run_id: str, signals: Iterable[DiagnosticSignal]) -> dict:
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics_version": DiagnosticsVersion,
        "diagnostics": [signal.to_dict() for signal in signals],
    }
