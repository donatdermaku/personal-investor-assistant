from __future__ import annotations

import pandas as pd
from datetime import date


def canonical_market_calendar(
    start: date,
    end: date,
    benchmark_prices: pd.DataFrame | None = None,
    ticker_prices: pd.DataFrame | None = None,
) -> tuple[pd.DatetimeIndex, str]:
    """
    Constructs the canonical market calendar (expected trading dates) between start and end (inclusive).

    Priority:
    1. Benchmark Calendar: If benchmark_prices is available and valid, use its dates.
    2. Union Tickers: If ticker_prices is available, use the union of all dates.
    3. Fallback: Naive business days (pd.bdate_range).

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        benchmark_prices: DataFrame containing benchmark prices usually with a "date" column.
        ticker_prices: DataFrame containing prices for all tickers, usually with a "date" column.

    Returns:
        tuple[pd.DatetimeIndex, str]:
            - The canonical calendar (DatetimeIndex), normalized to midnight, unique, sorted, filtered to [start, end].
            - The source identifier ("benchmark", "union_tickers", "bdate_range").
    """
    # Helper to clean and filter dates
    def _clean_dates(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        dt_index = pd.to_datetime(dates, errors="coerce")
        if isinstance(dt_index, pd.Series):
             dt_index = pd.DatetimeIndex(dt_index)
        # Normalize to midnight (remove time components)
        dt_index = dt_index.normalize()
        # Filter range
        mask = (dt_index >= pd.Timestamp(start)) & (dt_index <= pd.Timestamp(end))
        dt_index = dt_index[mask]
        # Unique and sort
        return dt_index.unique().sort_values()

    # 1. Benchmark Strategy
    if benchmark_prices is not None and not benchmark_prices.empty and "date" in benchmark_prices.columns:
        bench_dates = _clean_dates(benchmark_prices["date"])
        if not bench_dates.empty:
            return bench_dates, "benchmark"

    # 2. Union Tickers Strategy
    if ticker_prices is not None and not ticker_prices.empty and "date" in ticker_prices.columns:
        union_dates = _clean_dates(ticker_prices["date"])
        if not union_dates.empty:
            return union_dates, "union_tickers"

    # 3. Fallback Strategy
    fallback = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    fallback = fallback.normalize().unique().sort_values()
    return fallback, "bdate_range"
