"""
Market Data Validator

Validates market data integrity before storing in cache.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str
    failures: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.failures is None:
            self.failures = []


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    passed: bool
    results: List[ValidationResult]
    total_checks: int
    passed_checks: int
    failed_checks: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "results": [
                {
                    "passed": r.passed,
                    "message": r.message,
                    "failure_count": len(r.failures)
                }
                for r in self.results
            ]
        }


class MarketDataValidator:
    """Validator for market data quality checks."""
    
    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
    
    def validate_no_nulls(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check critical columns have no nulls.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            ValidationResult
        """
        if df.empty:
            return ValidationResult(
                passed=True,
                message="Empty DataFrame, skipping null check"
            )
        
        failures = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue
            
            null_count = df[col].isnull().sum()
            if null_count > 0:
                failures.append({
                    "column": col,
                    "null_count": int(null_count),
                    "total_rows": len(df)
                })
        
        if failures:
            return ValidationResult(
                passed=False,
                message=f"Null values found in {len(failures)} column(s)",
                failures=failures
            )
        
        return ValidationResult(
            passed=True,
            message="No nulls in critical columns"
        )
    
    def validate_ohlc_logic(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check OHLC logic: high >= low, close within [low, high].
        
        Args:
            df: DataFrame with market data
            
        Returns:
            ValidationResult
        """
        if df.empty:
            return ValidationResult(
                passed=True,
                message="Empty DataFrame, skipping OHLC check"
            )
        
        required = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required):
            return ValidationResult(
                passed=True,
                message="Missing OHLC columns, skipping check"
            )
        
        failures = []
        
        # Check high >= low
        invalid_hl = df[df["high"] < df["low"]]
        if not invalid_hl.empty:
            for idx, row in invalid_hl.iterrows():
                failures.append({
                    "type": "high_less_than_low",
                    "date": str(row.get("date", idx)),
                    "high": float(row["high"]),
                    "low": float(row["low"])
                })
        
        # Check close within [low, high]
        invalid_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
        if not invalid_close.empty:
            for idx, row in invalid_close.iterrows():
                failures.append({
                    "type": "close_out_of_range",
                    "date": str(row.get("date", idx)),
                    "close": float(row["close"]),
                    "low": float(row["low"]),
                    "high": float(row["high"])
                })
        
        if failures:
            return ValidationResult(
                passed=False,
                message=f"OHLC logic violations: {len(failures)} rows",
                failures=failures[:10]  # Limit to first 10
            )
        
        return ValidationResult(
            passed=True,
            message="OHLC logic valid"
        )
    
    def validate_positive_volume(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check volume > 0 (or allow 0 for some tickers).
        
        Args:
            df: DataFrame with market data
            
        Returns:
            ValidationResult
        """
        if df.empty or "volume" not in df.columns:
            return ValidationResult(
                passed=True,
                message="Empty DataFrame or no volume column"
            )
        
        # Allow volume = 0 but not negative
        negative_volume = df[df["volume"] < 0]
        
        if not negative_volume.empty:
            failures = []
            for idx, row in negative_volume.iterrows():
                failures.append({
                    "date": str(row.get("date", idx)),
                    "volume": float(row["volume"])
                })
            
            return ValidationResult(
                passed=False,
                message=f"Negative volume found in {len(failures)} rows",
                failures=failures[:10]
            )
        
        return ValidationResult(
            passed=True,
            message="Volume values valid"
        )
    
    def validate_no_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check no duplicate (ticker, date) pairs.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            ValidationResult
        """
        if df.empty:
            return ValidationResult(
                passed=True,
                message="Empty DataFrame, skipping duplicate check"
            )
        
        if "date" not in df.columns:
            return ValidationResult(
                passed=True,
                message="No date column, skipping duplicate check"
            )
        
        # Check for duplicates
        if "ticker" in df.columns:
            duplicates = df[df.duplicated(subset=["ticker", "date"], keep=False)]
        else:
            duplicates = df[df.duplicated(subset=["date"], keep=False)]
        
        if not duplicates.empty:
            failures = []
            for idx, row in duplicates.iterrows():
                failures.append({
                    "date": str(row["date"]),
                    "ticker": str(row.get("ticker", "N/A"))
                })
            
            return ValidationResult(
                passed=False,
                message=f"Duplicate records found: {len(duplicates)} rows",
                failures=failures[:10]
            )
        
        return ValidationResult(
            passed=True,
            message="No duplicate records"
        )
    
    def validate_date_format(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check date column is properly formatted.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            ValidationResult
        """
        if df.empty or "date" not in df.columns:
            return ValidationResult(
                passed=True,
                message="Empty DataFrame or no date column"
            )
        
        try:
            # Try to convert to datetime
            dates = pd.to_datetime(df["date"], errors="raise")
            
            # Check for NaT values
            nat_count = dates.isna().sum()
            if nat_count > 0:
                return ValidationResult(
                    passed=False,
                    message=f"Invalid dates found: {nat_count} NaT values",
                    failures=[{"nat_count": int(nat_count)}]
                )
            
            return ValidationResult(
                passed=True,
                message="Date format valid"
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Date parsing failed: {str(e)}",
                failures=[{"error": str(e)}]
            )
    
    def validate_all(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validations and return comprehensive report.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            ValidationReport
        """
        results = []
        
        # Run all validation checks
        results.append(self.validate_no_nulls(df))
        results.append(self.validate_ohlc_logic(df))
        results.append(self.validate_positive_volume(df))
        results.append(self.validate_no_duplicates(df))
        results.append(self.validate_date_format(df))
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        all_passed = failed_count == 0
        
        report = ValidationReport(
            passed=all_passed,
            results=results,
            total_checks=len(results),
            passed_checks=passed_count,
            failed_checks=failed_count
        )
        
        if not all_passed:
            logger.warning(
                "Market data validation failed",
                passed=passed_count,
                failed=failed_count,
                total=len(results)
            )
        
        return report
