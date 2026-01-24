from __future__ import annotations

import pandas as pd

from src.coverage import CoveragePolicy, build_coverage_summary


def _price_frame(ticker: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * len(dates),
            "close": [100.0] * len(dates),
            "adj_close": [100.0] * len(dates),
        }
    )


def test_coverage_unknown_without_prices() -> None:
    summary = build_coverage_summary(pd.DataFrame(), required_tickers=["AAPL"])
    assert summary["status"] == "unknown"
    assert "NO_PRICES" in summary["reason_codes"]


def test_coverage_insufficient_short_history() -> None:
    policy = CoveragePolicy(min_score_for_kpis=0.95, min_history_days=5, max_gap_days=1)
    prices = _price_frame("AAPL", ["2024-01-01", "2024-01-02"])
    summary = build_coverage_summary(
        prices,
        required_tickers=["AAPL"],
        as_of="2024-01-05",
        policy=policy,
    )
    assert summary["status"] == "insufficient"
    assert summary["per_ticker"]["AAPL"]["history_days"] == 2


def test_coverage_sufficient_history() -> None:
    policy = CoveragePolicy(min_score_for_kpis=0.8, min_history_days=5, max_gap_days=2)
    prices = _price_frame(
        "AAPL",
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )
    summary = build_coverage_summary(
        prices,
        required_tickers=["AAPL"],
        as_of="2024-01-05",
        policy=policy,
    )
    assert summary["status"] == "sufficient"
    assert summary["aggregate"]["min_ticker_score"] >= policy.min_score_for_kpis
