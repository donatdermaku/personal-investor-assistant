from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.analytics.contracts import KPI_DEPENDENCIES, evaluate_metric_status
from market_data.calendar import canonical_market_calendar


@dataclass(frozen=True)
class CoveragePolicy:
    min_score_for_kpis: float = 0.95
    min_history_days: int = 252
    max_gap_days: int = 5


def _as_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _largest_gap_days(expected_dates: list[date], actual_dates: set[date]) -> int:
    largest = 0
    current = 0
    for day in expected_dates:
        if day in actual_dates:
            largest = max(largest, current)
            current = 0
        else:
            current += 1
    largest = max(largest, current)
    return largest


def _score_ticker(
    dates: list[date],
    expected_dates: list[date],
    *,
    policy: CoveragePolicy,
) -> dict:
    if not dates:
        return {
            "score": 0.0,
            "history_days": 0,
            "missing_days": policy.min_history_days,
            "largest_gap_days": policy.min_history_days,
            "status": "missing",
            "reason_codes": ["NO_DATA"],
        }

    expected_set = set(expected_dates)
    actual_set = {d for d in dates if d in expected_set}
    history_days = len(actual_set)
    missing_days = max(0, len(expected_set) - history_days)
    largest_gap = _largest_gap_days(expected_dates, actual_set)

    # Coverage Completeness (Quality)
    # 0.0 if nothing expected (shouldn't happen if we call it right, but safe fallback)
    completeness = history_days / len(expected_dates) if expected_dates else 1.0
    
    if largest_gap <= policy.max_gap_days:
        gap_penalty = 1.0
    else:
        gap_penalty = max(0.0, 1.0 - ((largest_gap - policy.max_gap_days) / policy.max_gap_days))

    score = max(0.0, completeness * gap_penalty)
    
    reasons: list[str] = []
    is_short = history_days < policy.min_history_days
    if is_short:
        reasons.append("HISTORY_SHORT")
    if missing_days > 0:
        reasons.append("MISSING_DAYS")
    if largest_gap > policy.max_gap_days:
        reasons.append("GAPS_LARGE")

    # Status relies on both Quality (score) and Quantity (history_days)
    status = "ok" if (score >= policy.min_score_for_kpis and not is_short) else "low"
    
    return {
        "score": float(score),
        "history_days": int(history_days),
        "missing_days": int(missing_days),
        "largest_gap_days": int(largest_gap),
        "status": status,
        "reason_codes": reasons,
    }


def build_coverage_summary(
    prices: pd.DataFrame,
    *,
    required_tickers: list[str],
    benchmark_ticker: str | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    risk_free_series: pd.DataFrame | None = None,
    as_of: str | date | None = None,
    policy: CoveragePolicy | None = None,
    required_start_per_ticker: dict[str, date] | None = None,
) -> dict:
    policy = policy or CoveragePolicy()
    required_start_per_ticker = required_start_per_ticker or {}

    def metric_status_from_coverage(
        coverage: dict[str, dict[str, object]], 
        aggregate: dict[str, object] | None = None
    ) -> tuple[dict[str, str], dict[str, list[str]], dict[str, float]]:
        metric_status: dict[str, str] = {}
        metric_reasons: dict[str, list[str]] = {}
        metric_coverage: dict[str, float] = {}
        
        # Use provided aggregate or empty dict for early-return paths
        agg = aggregate or {}
        
        # Helper to get score for dependency
        def get_score(dep: str) -> float:
            if dep == "prices":
                 return agg.get("min_ticker_score", 0.0)
            if dep == "benchmark":
                 return agg.get("benchmark_score") or 0.0
            if dep == "risk_free":
                 return agg.get("rf_score") or 0.0
            return 0.0

        for key, deps in KPI_DEPENDENCIES.items():
            status, reasons = evaluate_metric_status(coverage, deps)
            
            # Map "available_low_coverage" to "active" status if needed, or just use as is.
            # The prompt says: "Metrics that previously computed but were insufficient now become available_low_coverage".
            # So status is "available_low_coverage".
            
            metric_status[key] = status
            metric_reasons[key] = reasons
            
            # Use score of first dependency for now
            primary_dep = deps[0] if deps else "prices"
            metric_coverage[key] = get_score(primary_dep)
            
        return metric_status, metric_reasons, metric_coverage

    if prices is None or prices.empty or "date" not in prices.columns or "ticker" not in prices.columns:
        coverage = {
            "prices": {"status": "insufficient", "reason_codes": ["NO_PRICES"]},
            "benchmark": {"status": "unknown", "reason_codes": []},
            "risk_free": {"status": "unknown", "reason_codes": []},
            "macro": {"status": "unknown", "reason_codes": []},
        }
        metric_status, metric_reasons, metric_coverage = metric_status_from_coverage(coverage)
        return {
            "as_of": _as_date(as_of).isoformat() if _as_date(as_of) else None,
            "status": "insufficient",
            "score": 0.0,
            "policy": {
                "min_score_for_kpis": policy.min_score_for_kpis,
                "min_history_days": policy.min_history_days,
                "max_gap_days": policy.max_gap_days,
            },
            "required": {
                "tickers": required_tickers,
                "history_days_needed": policy.min_history_days,
            },
            "per_ticker": {},
            "aggregate": {
                "coverage_ratio": 0.0,
                "min_ticker_score": 0.0,
                "benchmark_score": None,
                "rf_score": None,
            },
            "coverage": coverage,
            "metric_status": metric_status,
            "metric_reasons": metric_reasons,
            "metric_coverage": metric_coverage,
            "reason_codes": ["NO_PRICES"],
            "contract_version": "coverage_summary_v2",
            "version": "2.0",
        }

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
    prices["ticker"] = prices["ticker"].astype(str)

    inferred_as_of = _as_date(prices["date"].max())
    if benchmark_prices is not None and not benchmark_prices.empty and "date" in benchmark_prices.columns:
        bench_max = pd.to_datetime(benchmark_prices["date"], errors="coerce").dt.date.max()
        if bench_max and (not inferred_as_of or bench_max > inferred_as_of):
            inferred_as_of = bench_max
    as_of_date = _as_date(as_of) or inferred_as_of

    if not as_of_date:
        coverage = {
            "prices": {"status": "insufficient", "reason_codes": ["NO_DATES"]},
            "benchmark": {"status": "unknown", "reason_codes": []},
            "risk_free": {"status": "unknown", "reason_codes": []},
            "macro": {"status": "unknown", "reason_codes": []},
        }
        metric_status, metric_reasons, metric_coverage = metric_status_from_coverage(coverage)
        return {
            "as_of": None,
            "status": "unknown",
            "score": 0.0,
            "policy": {
                "min_score_for_kpis": policy.min_score_for_kpis,
                "min_history_days": policy.min_history_days,
                "max_gap_days": policy.max_gap_days,
            },
            "required": {
                "tickers": required_tickers,
                "history_days_needed": policy.min_history_days,
            },
            "per_ticker": {},
            "aggregate": {
                "coverage_ratio": 0.0,
                "min_ticker_score": 0.0,
                "benchmark_score": None,
                "rf_score": None,
            },
            "coverage": coverage,
            "metric_status": metric_status,
            "metric_reasons": metric_reasons,
            "metric_coverage": metric_coverage,
            "reason_codes": ["NO_DATES"],
            "contract_version": "coverage_summary_v2",
            "version": "2.0",
        }

    per_ticker: dict[str, dict] = {}
    core_scores: list[float] = []

    # Calculate Canonical Calendar
    # Start date = max(as_of - min_history_days, earliest_date_needed)
    # Ideally we just look back min_history_days business days from as_of
    start_date = (pd.Timestamp(as_of_date) - pd.Timedelta(days=policy.min_history_days * 2)).date() # Approximation for start, calendar will filter
    
    # Better approach: Use naive window to establish RANGE, then refine with canonical
    naive_expected = pd.bdate_range(end=pd.Timestamp(as_of_date), periods=policy.min_history_days)
    start_date = naive_expected.min().date()

    expected_index, calendar_source = canonical_market_calendar(
        start=start_date,
        end=as_of_date,
        benchmark_prices=benchmark_prices,
        ticker_prices=prices,
    )
    expected_dates_list = [d.date() for d in expected_index]

    per_ticker: dict[str, dict] = {}
    core_scores: list[float] = []

    for ticker in required_tickers:
        ticker_dates = prices.loc[prices["ticker"] == ticker, "date"].dropna().unique().tolist()
        
        # Apply required start date constraint
        ticker_expected = expected_dates_list
        req_start = required_start_per_ticker.get(ticker)
        if req_start:
             ticker_expected = [d for d in expected_dates_list if d >= req_start]
        
        # If the required window is empty (e.g. required start > as_of), use empty list to avoid errors, 
        # or maybe we should default to at least one day? 
        # For now, if ticker_expected is empty, _score_ticker handles it (returns missing or 0 score).
        
        result = _score_ticker(ticker_dates, ticker_expected, policy=policy)
        per_ticker[ticker] = result
        per_ticker[ticker]["required_start"] = req_start.isoformat() if req_start else None
        
        core_scores.append(result["score"])

    benchmark_score = None
    if benchmark_ticker:
        bench_df = benchmark_prices if benchmark_prices is not None else pd.DataFrame()
        bench_dates = []
        if not bench_df.empty and "date" in bench_df.columns:
            bench_dates = pd.to_datetime(bench_df["date"], errors="coerce").dt.date.dropna().unique().tolist()
        
        # Benchmark must also cover the canonical calendar
        bench_result = _score_ticker(bench_dates, expected_dates_list, policy=policy)
        per_ticker[benchmark_ticker] = bench_result
        benchmark_score = bench_result["score"]

    status = "unknown"
    reason_codes: list[str] = []
    prices_status = "unknown"
    prices_reasons: list[str] = []
    if not core_scores:
        reason_codes.append("NO_CORE_TICKERS")
        prices_status = "insufficient"
        prices_reasons.append("NO_CORE_TICKERS")
    else:
        score = min(core_scores)
        if score >= policy.min_score_for_kpis:
            status = "sufficient"
            reason_codes.append("OK")
            prices_status = "sufficient"
        else:
            status = "insufficient"
            reason_codes.append("CORE_TICKER_INSUFFICIENT")
            prices_status = "insufficient"
            prices_reasons.append("CORE_TICKER_INSUFFICIENT")

    benchmark_status = "unknown"
    benchmark_reasons: list[str] = []
    if benchmark_ticker:
        if benchmark_score is None:
            benchmark_status = "insufficient"
            benchmark_reasons.append("BENCHMARK_MISSING")
        elif benchmark_score >= policy.min_score_for_kpis:
            benchmark_status = "sufficient"
        else:
            benchmark_status = "insufficient"
            benchmark_reasons.append("BENCHMARK_INSUFFICIENT")
            reason_codes.append("BENCHMARK_INSUFFICIENT")
    else:
        benchmark_status = "insufficient"
        benchmark_reasons.append("BENCHMARK_MISSING")

    risk_free_status = "unknown"
    risk_free_reasons: list[str] = []
    if risk_free_series is None or risk_free_series.empty:
        risk_free_status = "insufficient"
        risk_free_reasons.append("RF_MISSING")
        reason_codes.append("RF_MISSING")

    rf_score = None
    if risk_free_series is not None and not risk_free_series.empty and "date" in risk_free_series.columns:
        rf_dates = pd.to_datetime(risk_free_series["date"], errors="coerce").dt.date.dropna().unique().tolist()
        # Risk free must also cover the canonical calendar
        rf_result = _score_ticker(rf_dates, expected_dates_list, policy=policy)
        rf_score = rf_result["score"]
        if rf_score >= policy.min_score_for_kpis:
            risk_free_status = "sufficient"
        else:
            risk_free_status = "insufficient"
            risk_free_reasons.append("RF_INSUFFICIENT")

    aggregate = {
        "coverage_ratio": float(sum(core_scores) / len(core_scores)) if core_scores else 0.0,
        "min_ticker_score": float(min(core_scores)) if core_scores else 0.0,
        "benchmark_score": benchmark_score,
        "rf_score": rf_score,
        "calendar_source": calendar_source,
        "calendar_start": start_date.isoformat(),
        "calendar_end": as_of_date.isoformat(),
        "calendar_count": len(expected_dates_list),
        "required_start_per_ticker": {t: d.isoformat() for t, d in required_start_per_ticker.items()},
    }
    coverage = {
        "prices": {"status": prices_status, "reason_codes": prices_reasons},
        "benchmark": {"status": benchmark_status, "reason_codes": benchmark_reasons},
        "risk_free": {"status": risk_free_status, "reason_codes": risk_free_reasons},
        "macro": {"status": "unknown", "reason_codes": []},
    }
    metric_status, metric_reasons, metric_coverage = metric_status_from_coverage(coverage, aggregate)

    return {
        "as_of": as_of_date.isoformat(),
        "status": status,
        "score": aggregate["min_ticker_score"],
        "policy": {
            "min_score_for_kpis": policy.min_score_for_kpis,
            "min_history_days": policy.min_history_days,
            "max_gap_days": policy.max_gap_days,
        },
        "required": {
            "tickers": required_tickers + ([benchmark_ticker] if benchmark_ticker else []),
            "history_days_needed": policy.min_history_days,
        },
        "per_ticker": per_ticker,
        "aggregate": aggregate,
        "coverage": coverage,
        "metric_status": metric_status,
        "metric_reasons": metric_reasons,
        "metric_coverage": metric_coverage,
        "reason_codes": reason_codes,
        "contract_version": "coverage_summary_v2",
        "version": "2.0",
    }
