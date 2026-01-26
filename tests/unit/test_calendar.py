from datetime import date
import pandas as pd
import pytest
from market_data.calendar import canonical_market_calendar

def test_canonical_calendar_benchmark_priority():
    start = date(2023, 1, 1)
    end = date(2023, 1, 10)
    
    # Bench has 2023-01-03, 2023-01-04
    bench = pd.DataFrame({
        "date": [date(2023, 1, 3), date(2023, 1, 4)],
        "close": [100, 101]
    })
    
    # Ticker has 2023-01-03, 2023-01-05
    ticker = pd.DataFrame({
        "date": [date(2023, 1, 3), date(2023, 1, 5)],
        "close": [10, 11]
    })
    
    cal, source = canonical_market_calendar(start, end, benchmark_prices=bench, ticker_prices=ticker)
    
    assert source == "benchmark"
    assert len(cal) == 2
    assert cal[0].date() == date(2023, 1, 3)
    assert cal[1].date() == date(2023, 1, 4)

def test_canonical_calendar_union_fallback():
    start = date(2023, 1, 1)
    end = date(2023, 1, 10)
    
    # No benchmark
    bench = pd.DataFrame()
    
    # Ticker has 2023-01-03
    ticker = pd.DataFrame({
        "date": [date(2023, 1, 3)],
        "close": [10]
    })
    
    cal, source = canonical_market_calendar(start, end, benchmark_prices=bench, ticker_prices=ticker)
    
    assert source == "union_tickers"
    assert len(cal) == 1
    assert cal[0].date() == date(2023, 1, 3)

def test_canonical_calendar_bdate_fallback():
    start = date(2023, 1, 1)
    end = date(2023, 1, 5) # Sun to Thu
    
    cal, source = canonical_market_calendar(start, end)
    
    assert source == "bdate_range"
    # 2023-01-02 (Mon), 03 (Tue), 04 (Wed), 05 (Thu)
    expected = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)]
    assert len(cal) == 4
    for i, d in enumerate(expected):
        assert cal[i].date() == d
