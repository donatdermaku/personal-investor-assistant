"""
Rate limiter and retry logic for Yahoo Finance API calls.

Implements:
- Global token bucket rate limiter (1 req/1.5s, concurrency=1)
- Retry-After aware retry wrapper with exponential backoff
- Cache validation to prevent poisoned cache

Usage:
    from market_data.rate_limiter import throttled_fetch, validate_price_cache

    df = throttled_fetch(lambda: yf.download(...))
    if validate_price_cache(df, "2010-01-01", "2026-01-26"):
        # Safe to cache
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RateLimiter:
    """Simple token bucket rate limiter for Yahoo Finance calls.
    
    Thread-safe, process-wide limiter. All Yahoo calls should go through this.
    """
    requests_per_second: float = 0.67  # ~1 request per 1.5 seconds
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_request: float = 0.0

    def wait(self) -> None:
        """Block until rate limit allows next request."""
        with self._lock:
            now = time.time()
            min_interval = 1.0 / self.requests_per_second
            wait_time = max(0, self._last_request + min_interval - now)
            if wait_time > 0:
                # Add small jitter (50-150ms) to avoid bot-like patterns
                jitter = random.uniform(0.05, 0.15)
                time.sleep(wait_time + jitter)
            self._last_request = time.time()


# Global limiter - all Yahoo calls go through this
_yahoo_limiter = RateLimiter()


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_range: tuple[float, float] = (0.1, 0.3)


def _parse_retry_after(response_or_exception) -> float | None:
    """Extract Retry-After header value if present."""
    # yfinance doesn't expose headers directly, but if we catch HTTPError
    # we can try to extract it
    try:
        if hasattr(response_or_exception, "response"):
            resp = response_or_exception.response
            if resp is not None and hasattr(resp, "headers"):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        return float(retry_after)
                    except ValueError:
                        # Could be HTTP-date format, try parsing
                        pass
    except Exception:
        pass
    return None


def throttled_fetch(
    fetch_fn: Callable[[], T],
    *,
    config: RetryConfig | None = None,
    operation_name: str = "Yahoo fetch",
) -> T:
    """Execute a fetch function with rate limiting and retry logic.
    
    Args:
        fetch_fn: The function to call (should make the Yahoo API call)
        config: Retry configuration (uses defaults if None)
        operation_name: Name for logging purposes
        
    Returns:
        Result of fetch_fn
        
    Raises:
        Exception: If all retries exhausted
    """
    config = config or RetryConfig()
    last_exception = None
    backoff = config.initial_backoff_seconds
    
    for attempt in range(config.max_retries + 1):
        # Wait for rate limiter
        _yahoo_limiter.wait()
        
        try:
            result = fetch_fn()
            if attempt > 0:
                logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
            return result
            
        except Exception as exc:
            last_exception = exc
            error_str = str(exc).lower()
            
            # Check if it's a rate limit error (429)
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "too many requests" in error_str
            
            # Check if it's a server error (5xx)
            is_server_error = any(code in error_str for code in ["500", "502", "503", "504"])
            
            # Check for timeout
            is_timeout = "timeout" in error_str or "timed out" in error_str
            
            # Don't retry client errors (4xx) other than 429
            is_client_error = "400" in error_str or "401" in error_str or "403" in error_str or "404" in error_str
            
            if is_client_error and not is_rate_limit:
                logger.warning(f"{operation_name} failed with client error, not retrying: {exc}")
                raise
            
            if attempt >= config.max_retries:
                logger.error(f"{operation_name} failed after {config.max_retries + 1} attempts: {exc}")
                raise
            
            # Determine wait time
            if is_rate_limit:
                # Check for Retry-After header
                retry_after = _parse_retry_after(exc)
                if retry_after:
                    wait_time = retry_after + random.uniform(*config.jitter_range)
                    logger.warning(f"{operation_name} rate limited, waiting {wait_time:.1f}s (Retry-After)")
                else:
                    # Use longer backoff for rate limits
                    wait_time = min(backoff * 2, config.max_backoff_seconds)
                    logger.warning(f"{operation_name} rate limited, waiting {wait_time:.1f}s (backoff)")
            elif is_server_error or is_timeout:
                wait_time = min(backoff, config.max_backoff_seconds)
                logger.warning(f"{operation_name} server error/timeout, retrying in {wait_time:.1f}s")
            else:
                wait_time = backoff
                logger.warning(f"{operation_name} failed, retrying in {wait_time:.1f}s: {exc}")
            
            time.sleep(wait_time)
            backoff = min(backoff * config.backoff_multiplier, config.max_backoff_seconds)
    
    raise last_exception  # Should never reach here, but just in case


def validate_price_cache(
    df: pd.DataFrame,
    required_start: str,
    required_end: str,
    *,
    min_rows: int = 1000,
    required_columns: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate a price DataFrame before caching to prevent poisoned cache.
    
    Args:
        df: Price DataFrame to validate
        required_start: Required start date (YYYY-MM-DD)
        required_end: Required end date (YYYY-MM-DD)
        min_rows: Minimum expected rows for 2010+ daily data
        required_columns: Required column names
        
    Returns:
        Tuple of (is_valid, list of reason codes)
    """
    if required_columns is None:
        required_columns = ["date", "close", "adj_close"]
    
    reasons: list[str] = []
    
    # Check if DataFrame is empty
    if df is None or df.empty:
        return False, ["EMPTY_DATAFRAME"]
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        reasons.append(f"MISSING_COLUMNS:{','.join(missing_cols)}")
    
    # Check date column
    if "date" not in df.columns:
        return False, reasons + ["NO_DATE_COLUMN"]
    
    dates = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = dates.dropna()
    
    if valid_dates.empty:
        return False, reasons + ["NO_VALID_DATES"]
    
    # Check date range coverage
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    
    try:
        req_start_date = datetime.strptime(required_start, "%Y-%m-%d").date()
        req_end_date = datetime.strptime(required_end, "%Y-%m-%d").date()
    except ValueError as e:
        return False, reasons + [f"INVALID_DATE_FORMAT:{e}"]
    
    # Allow small tolerance for start date (holidays/weekends around required start).
    start_tolerance = timedelta(days=7)
    if min_date > (req_start_date + start_tolerance):
        # Only fail on start gaps when the frame is undersized; this allows IPO-era data
        # to cache while still blocking tiny/poisoned frames.
        if len(df) < min_rows:
            reasons.append(f"START_NOT_COVERED:need={required_start},got={min_date.isoformat()}")
    
    # Allow 5 business days tolerance for end date (weekends, holidays)
    end_tolerance = timedelta(days=7)
    if max_date < (req_end_date - end_tolerance):
        reasons.append(f"END_NOT_COVERED:need={required_end},got={max_date.isoformat()}")
    
    # Check row count - make it proportional to the actual data range
    # This allows IPO-era stocks (started after 2010) to cache while blocking tiny/poisoned frames
    actual_days = (max_date - min_date).days
    expected_trading_days = max(1, actual_days * 5 // 7)  # Rough estimate accounting for weekends
    # Require at least 70% of expected trading days, with minimum of 50 rows
    min_expected = max(50, int(expected_trading_days * 0.7))
    if len(df) < min_expected:
        reasons.append(f"TOO_FEW_ROWS:expected>={min_expected},got={len(df)}")
    
    # Check for duplicate dates
    if valid_dates.duplicated().any():
        dup_count = valid_dates.duplicated().sum()
        reasons.append(f"DUPLICATE_DATES:{dup_count}")
    
    # Check that dates are sorted (not strictly required but good practice)
    if not valid_dates.is_monotonic_increasing:
        reasons.append("DATES_NOT_SORTED")
    
    is_valid = len(reasons) == 0
    return is_valid, reasons


def get_rate_limiter() -> RateLimiter:
    """Get the global Yahoo rate limiter (for testing/debugging)."""
    return _yahoo_limiter
