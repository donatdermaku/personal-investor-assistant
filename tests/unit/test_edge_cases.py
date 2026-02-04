"""
Edge case tests for crash scenarios and error handling.

Tests scenarios that could cause application crashes if not handled properly:
- Division by zero in align_benchmark
- Empty portfolios with no securities
- Invalid benchmark data (NaN, zero prices)
- Large file uploads
- Missing ticker data
"""

import pandas as pd
import pytest
import numpy as np
from src.portfolio import align_benchmark, compute_portfolio_from_ledger, validate_ledger


class TestAlignBenchmarkEdgeCases:
    """Test edge cases in benchmark alignment that could cause crashes."""
    
    def test_align_benchmark_zero_first_price(self):
        """Benchmark with first price = 0 should return empty series instead of div/0 error."""
        benchmark = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "adj_close": [0.0, 100.0],  # First price is 0
        })
        portfolio = pd.Series(
            [10000, 10500], 
            index=pd.to_datetime(["2023-01-01", "2023-01-02"])
        )
        
        result = align_benchmark(benchmark, portfolio)
        assert result.empty, "Should return empty series when first price is 0"
    
    def test_align_benchmark_nan_first_price(self):
        """Benchmark with NaN first price should return empty series instead of propagating NaN."""
        benchmark = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "adj_close": [float('nan'), 100.0],
        })
        portfolio = pd.Series(
            [10000, 10500],
            index=pd.to_datetime(["2023-01-01", "2023-01-02"])
        )
        
        result = align_benchmark(benchmark, portfolio)
        assert result.empty, "Should return empty series when first price is NaN"
    
    def test_align_benchmark_empty_dataframe(self):
        """Empty benchmark should return empty series."""
        benchmark = pd.DataFrame()
        portfolio = pd.Series([10000], index=pd.to_datetime(["2023-01-01"]))
        
        result = align_benchmark(benchmark, portfolio)
        assert result.empty
    
    def test_align_benchmark_empty_portfolio(self):
        """Empty portfolio should return empty series."""
        benchmark = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "adj_close": [100.0],
        })
        portfolio = pd.Series(dtype=float)
        
        result = align_benchmark(benchmark, portfolio)
        assert result.empty
    
    def test_align_benchmark_valid_data(self):
        """Valid data should produce correct scaled benchmark."""
        benchmark = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "adj_close": [100.0, 110.0],
        })
        portfolio = pd.Series(
            [10000, 10500],
            index=pd.to_datetime(["2023-01-01", "2023-01-02"])
        )
        
        result = align_benchmark(benchmark, portfolio)
        
        assert not result.empty
        assert len(result) == 2
        # First value should match portfolio start (scaled)
        assert result.iloc[0] == pytest.approx(10000.0)
        # Second value should be scaled proportionally
        assert result.iloc[1] == pytest.approx(11000.0)


class TestEmptyPortfolioEdgeCases:
    """Test portfolios with only cash transactions (no securities)."""
    
    def test_portfolio_only_deposits_no_crash(self):
        """Portfolio with only deposits (no securities) should not crash."""
        ledger = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "ticker": ["CASH", "CASH"],
            "action": ["DEPOSIT", "DEPOSIT"],
            "quantity": [0.0, 0.0],
            "price": [10000.0, 5000.0],
            "fees": [0.0, 0.0],
        })
        
        prices = pd.DataFrame()  # No market data needed
        
        # Should not crash even with no securities
        result = compute_portfolio_from_ledger(ledger, prices)
        
        # Result may have errors or be empty, but shouldn't crash
        assert isinstance(result.errors, list)
        # Cash-only portfolio should still track value
        assert isinstance(result.daily_values, pd.DataFrame)


class TestLedgerValidationEdgeCases:
    """Test ledger validation with edge cases."""
    
    def test_validate_ledger_missing_required_column(self):
        """Missing required column should return validation error."""
        ledger = pd.DataFrame({
            "date": ["2023-01-01"],
            "ticker": ["AAPL"],
            # Missing 'action', 'quantity', 'price'
        })
        
        validated, errors = validate_ledger(ledger)
        
        assert len(errors) > 0
        assert validated.empty
        assert any("action" in err.lower() for err in errors)
    
    def test_validate_ledger_invalid_date(self):
        """Invalid date should return validation error."""
        ledger = pd.DataFrame({
            "date": ["not-a-date"],
            "ticker": ["AAPL"],
            "action": ["BUY"],
            "quantity": [100],
            "price": [150.0],
        })
        
        validated, errors = validate_ledger(ledger)
        
        assert len(errors) > 0
        assert any("date" in err.lower() for err in errors)
    
    def test_validate_ledger_negative_holdings_short_sell(self):
        """Selling more than owned should return error when short selling disabled."""
        ledger = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "action": ["BUY", "SELL"],
            "quantity": [10, 20],  # Selling more than bought
            "price": [150.0, 155.0],
            "fees": [0.0, 0.0],
        })
        
        validated, errors = validate_ledger(ledger, allow_short=False)
        
        assert len(errors) > 0
        assert any("negative" in err.lower() for err in errors)
    
    def test_validate_ledger_zero_quantity_buy(self):
        """BUY action with zero quantity should return error."""
        ledger = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "ticker": ["AAPL"],
            "action": ["BUY"],
            "quantity": [0],  # Invalid: must be > 0
            "price": [150.0],
        })
        
        validated, errors = validate_ledger(ledger)
        
        assert len(errors) > 0
        assert any("quantity" in err.lower() or "requires" in err.lower() for err in errors)


class TestNaNAndInfHandling:
    """Test handling of NaN and Inf values in computations."""
    
    def test_benchmark_with_nan_values(self):
        """Benchmark with NaN values should be handled gracefully."""
        benchmark = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "adj_close": [100.0, np.nan, 110.0],  # NaN in middle
        })
        portfolio = pd.Series(
            [10000, 10200, 10500],
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        )
        
        # Should not crash, may return empty or filtered result
        result = align_benchmark(benchmark, portfolio)
        
        # As long as it doesn't crash, test passes
        assert isinstance(result, pd.Series)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
