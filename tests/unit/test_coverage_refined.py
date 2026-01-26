from datetime import date
import pandas as pd
import pytest
from src.coverage import CoveragePolicy, _score_ticker

def test_score_ticker_perfect_match():
    policy = CoveragePolicy(min_history_days=5)
    dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
    expected = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
    
    result = _score_ticker(dates, expected, policy=policy)
    # With min_history_days=5, score is penalized even if we have all expected 3 dates
    # history_score = 3/5 = 0.6
    # missing_days = 2 -> penalty = 1 - 0.4 = 0.6
    # score = 0.36
    assert result["status"] == "low"


def test_score_ticker_missing_dates():
    policy = CoveragePolicy(min_history_days=5)
    # Expected 5 days, got 3
    expected = [date(2023, 1, i) for i in range(1, 6)]
    dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
    
    result = _score_ticker(dates, expected, policy=policy)
    # 3/5 history score = 0.6
    # missing 2 days -> penalty 1 - (2/5) = 0.6
    # score = 0.6 * 0.6 = 0.36
    assert result["score"] < 1.0
    assert "MISSING_DAYS" in result["reason_codes"]
