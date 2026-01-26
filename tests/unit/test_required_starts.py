from datetime import date
import pandas as pd
import pytest
from src.coverage import CoveragePolicy, _score_ticker
from src.analytics.required_start import compute_required_start_per_ticker

def test_compute_required_start_basic():
    holdings = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "date": [date(2023, 1, 10), date(2023, 1, 11), date(2023, 1, 5), date(2023, 1, 6)],
        "quantity": [10, 10, 100, 100]
    })
    starts = compute_required_start_per_ticker(holdings)
    assert starts["AAPL"] == date(2023, 1, 10)
    assert starts["MSFT"] == date(2023, 1, 5)

def test_compute_required_start_ignore_zero():
    holdings = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "date": [date(2023, 1, 1), date(2023, 1, 2)],
        "quantity": [0, 10]
    })
    starts = compute_required_start_per_ticker(holdings)
    assert starts["AAPL"] == date(2023, 1, 2)

def test_score_ticker_short_but_perfect_coverage():
    policy = CoveragePolicy(min_history_days=5)
    
    dates = [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)]
    # We only expect these 3 days because we filtered by required_start
    expected_dates = [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)]
    
    result = _score_ticker(dates, expected_dates, policy=policy)
    
    # Coverage score should be 1.0 (3/3)
    assert result["score"] == 1.0
    
    # Status should be "low" because 3 < 5
    assert result["status"] == "low"
    assert "HISTORY_SHORT" in result["reason_codes"]
