from __future__ import annotations

from typing import Iterable

from src.diagnostics.contracts import DiagnosticSignal, as_of_label


def _metric_sufficient(coverage_summary: dict | None, metric_key: str) -> bool:
    if not coverage_summary:
        return True
    status = coverage_summary.get("metric_status", {}).get(metric_key)
    return status != "insufficient"


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def concentration_risk(
    contributions: Iterable[dict],
    *,
    as_of: str | None,
    coverage_summary: dict | None,
    top_n: int = 3,
    medium_threshold: float = 0.50,
    high_threshold: float = 0.70,
) -> DiagnosticSignal | None:
    if not _metric_sufficient(coverage_summary, "volatility"):
        return None
    rows = [row for row in contributions if row.get("ticker") not in (None, "CASH")]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: _safe_float(r.get("volatility_pct")) or 0.0, reverse=True)
    top = rows[:top_n]
    total = sum(_safe_float(row.get("volatility_pct")) or 0.0 for row in top)
    if total < medium_threshold:
        return None
    severity = "high" if total >= high_threshold else "medium"
    evidence = [f"{row.get('ticker')}: {(((_safe_float(row.get('volatility_pct')) or 0.0) * 100)):.1f}% vol"
                for row in top]
    return DiagnosticSignal(
        key="CONCENTRATION_RISK",
        category="risk",
        severity=severity,
        summary="Volatility concentrated in top holdings",
        evidence=evidence,
        metrics_used=["volatility_pct"],
        as_of=as_of,
        confidence=min(1.0, total),
    )


def single_asset_dominance(
    weights: dict[str, float],
    *,
    as_of: str | None,
    coverage_summary: dict | None,
    medium_threshold: float = 0.25,
    high_threshold: float = 0.40,
) -> DiagnosticSignal | None:
    if not _metric_sufficient(coverage_summary, "portfolio_value"):
        return None
    if not weights:
        return None
    filtered = {k: v for k, v in weights.items() if k and k != "CASH"}
    if not filtered:
        return None
    ticker, weight = max(filtered.items(), key=lambda item: item[1])
    if weight < medium_threshold:
        return None
    severity = "high" if weight >= high_threshold else "medium"
    return DiagnosticSignal(
        key="SINGLE_ASSET_DOMINANCE",
        category="structure",
        severity=severity,
        summary="Largest holding dominates portfolio weight",
        evidence=[f"{ticker}: {weight * 100:.1f}% weight"],
        metrics_used=["holding_weight"],
        as_of=as_of,
        confidence=min(1.0, weight),
    )


def drawdown_sensitivity(
    portfolio_drawdown: float | None,
    benchmark_drawdown: float | None,
    *,
    as_of: str | None,
    coverage_summary: dict | None,
    medium_gap: float = 0.05,
    high_gap: float = 0.10,
) -> DiagnosticSignal | None:
    if not _metric_sufficient(coverage_summary, "max_drawdown"):
        return None
    if not _metric_sufficient(coverage_summary, "benchmark_correlation"):
        return None
    if portfolio_drawdown is None or benchmark_drawdown is None:
        return None
    gap = abs(portfolio_drawdown) - abs(benchmark_drawdown)
    if gap < medium_gap:
        return None
    severity = "high" if gap >= high_gap else "medium"
    return DiagnosticSignal(
        key="DRAWNDOWN_SENSITIVITY",
        category="risk",
        severity=severity,
        summary="Portfolio drawdown exceeds benchmark",
        evidence=[
            f"Portfolio max DD: {portfolio_drawdown * 100:.1f}%",
            f"Benchmark max DD: {benchmark_drawdown * 100:.1f}%",
        ],
        metrics_used=["max_drawdown", "benchmark_drawdown"],
        as_of=as_of,
        confidence=min(1.0, gap / high_gap) if high_gap else 0.0,
    )


def return_driver_mismatch(
    allocation: float | None,
    selection: float | None,
    total_return: float | None,
    *,
    as_of: str | None,
    coverage_summary: dict | None,
) -> DiagnosticSignal | None:
    if not _metric_sufficient(coverage_summary, "allocation_effect"):
        return None
    if allocation is None or selection is None or total_return is None:
        return None
    if total_return == 0:
        return None
    allocation_share = abs(allocation) / max(abs(total_return), 1e-9)
    if allocation_share < 0.6 or selection >= 0:
        return None
    severity = "high" if allocation_share >= 0.8 else "medium"
    return DiagnosticSignal(
        key="RETURN_DRIVER_MISMATCH",
        category="performance",
        severity=severity,
        summary="Returns driven by allocation while selection detracts",
        evidence=[
            f"Allocation: {allocation * 100:.1f}%",
            f"Selection: {selection * 100:.1f}%",
        ],
        metrics_used=["allocation_effect", "selection_effect"],
        as_of=as_of,
        confidence=min(1.0, allocation_share),
    )


def risk_free_dependency_warning(
    coverage_summary: dict | None,
    *,
    as_of: str | None,
) -> DiagnosticSignal | None:
    if not coverage_summary:
        return None
    sharpe_status = coverage_summary.get("metric_status", {}).get("sharpe")
    twr_status = coverage_summary.get("metric_status", {}).get("twr")
    sharpe_reasons = coverage_summary.get("metric_reasons", {}).get("sharpe", [])
    if sharpe_status != "insufficient" or twr_status == "insufficient":
        return None
    if "RF_MISSING" not in sharpe_reasons:
        return None
    return DiagnosticSignal(
        key="RISK_FREE_DEPENDENCY_WARNING",
        category="data",
        severity="low",
        summary="Sharpe unavailable due to missing risk-free rate",
        evidence=["Risk-free series missing (RF_MISSING)."],
        metrics_used=["sharpe", "risk_free_series"],
        as_of=as_of,
        confidence=0.6,
    )


def benchmark_mismatch(
    tracking_error: float | None,
    correlation: float | None,
    *,
    as_of: str | None,
    coverage_summary: dict | None,
    low_tracking_error: float = 0.02,
    low_correlation: float = 0.6,
) -> DiagnosticSignal | None:
    if not _metric_sufficient(coverage_summary, "tracking_error"):
        return None
    if not _metric_sufficient(coverage_summary, "benchmark_correlation"):
        return None
    if tracking_error is None and correlation is None:
        return None
    triggered = False
    evidence: list[str] = []
    if tracking_error is not None and tracking_error < low_tracking_error:
        triggered = True
        evidence.append(f"Tracking error: {tracking_error * 100:.2f}%")
    if correlation is not None and correlation < low_correlation:
        triggered = True
        evidence.append(f"Correlation: {correlation:.2f}")
    if not triggered:
        return None
    return DiagnosticSignal(
        key="BENCHMARK_MISMATCH",
        category="performance",
        severity="medium",
        summary="Benchmark relationship looks weak",
        evidence=evidence,
        metrics_used=["tracking_error", "benchmark_correlation"],
        as_of=as_of,
        confidence=0.5,
    )


def short_history_warning(
    coverage_summary: dict | None,
    *,
    as_of: str | None,
) -> DiagnosticSignal | None:
    if not coverage_summary:
        return None
    policy = coverage_summary.get("policy", {})
    min_days = policy.get("min_history_days")
    per_ticker = coverage_summary.get("per_ticker", {})
    if not per_ticker or not min_days:
        return None
    short = [ticker for ticker, meta in per_ticker.items() if meta.get("history_days", min_days) < min_days]
    if not short:
        return None
    evidence = [f"{ticker}: {per_ticker[ticker].get('history_days')} days" for ticker in short]
    return DiagnosticSignal(
        key="SHORT_HISTORY_WARNING",
        category="data",
        severity="low",
        summary="Insufficient history for rolling metrics",
        evidence=evidence,
        metrics_used=["rolling_metrics"],
        as_of=as_of,
        confidence=0.4,
    )


def normalize_signals(signals: Iterable[DiagnosticSignal]) -> list[DiagnosticSignal]:
    return [signal for signal in signals if signal is not None]
