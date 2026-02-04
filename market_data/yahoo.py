from __future__ import annotations

from datetime import date
from typing import Tuple

import pandas as pd
import yfinance as yf

from market_data.contracts import MarketDataError, validate_price_frame
from market_data.rate_limiter import throttled_fetch


def _normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    import logging
    logger = logging.getLogger(__name__)
    
    if raw is None or raw.empty:
        logger.warning("Yahoo Finance returned empty dataframe")
        return pd.DataFrame()
    
    df = raw.copy()
    logger.debug(f"Raw Yahoo columns: {df.columns.tolist()}, shape: {df.shape}")
    
    # Try to get date from index first
    if "Date" not in df.columns:
        if df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        else:
            logger.error(f"Cannot find date column in Yahoo data. Columns: {df.columns.tolist()}, Index: {df.index.name}")
            # Return empty instead of corrupted data
            return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance often returns (field, ticker); drop ticker level to keep field names.
        if df.columns.nlevels >= 2:
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        cols = {c: str(c).strip() for c in df.columns}
        df = df.rename(columns=cols)
    
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    
    df.columns = [c.lower() for c in df.columns]
    
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    
    logger.debug(f"Normalized columns: {df.columns.tolist()}")
    
    # Final validation: ensure date column exists
    if "date" not in df.columns:
        logger.error("Date column missing after normalization! This should not happen.")
        return pd.DataFrame()
    
    return df


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch price data from Yahoo Finance with rate limiting and retry.
    
    Uses global rate limiter to prevent 429 errors and implements
    exponential backoff with Retry-After header support.
    """
    def _do_fetch() -> pd.DataFrame:
        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        return raw
    
    try:
        raw = throttled_fetch(
            _do_fetch,
            operation_name=f"fetch_prices({ticker})",
        )
    except Exception as exc:
        raise MarketDataError(
            error_code="MARKET_DATA_FETCH_FAILED",
            message=f"Failed to fetch prices for {ticker}.",
            details={"ticker": ticker, "error": str(exc)},
            hint="Retry later or verify the ticker symbol.",
        )
    df = _normalize_prices(raw)
    df["ticker"] = ticker
    return validate_price_frame(df, ticker)


def fetch_dividends_and_splits(
    ticker: str,
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch dividend and split data from Yahoo Finance with rate limiting."""
    def _do_fetch():
        yf_ticker = yf.Ticker(ticker)
        return yf_ticker.dividends, yf_ticker.splits
    
    try:
        dividends, splits = throttled_fetch(
            _do_fetch,
            operation_name=f"fetch_dividends_and_splits({ticker})",
        )
    except Exception as exc:
        raise MarketDataError(
            error_code="MARKET_DATA_FETCH_FAILED",
            message=f"Failed to fetch corporate actions for {ticker}.",
            details={"ticker": ticker, "error": str(exc)},
            hint="Retry later or verify the ticker symbol.",
        )

    div_df = dividends.reset_index().rename(columns={"Date": "date", "Dividends": "amount"})
    div_df["date"] = pd.to_datetime(div_df["date"], errors="coerce").dt.date
    div_df = div_df[(div_df["date"] >= date.fromisoformat(start)) & (div_df["date"] <= date.fromisoformat(end))]

    splits_df = splits.reset_index().rename(columns={"Date": "date", "Stock Splits": "ratio"})
    splits_df["date"] = pd.to_datetime(splits_df["date"], errors="coerce").dt.date
    splits_df = splits_df[(splits_df["date"] >= date.fromisoformat(start)) & (splits_df["date"] <= date.fromisoformat(end))]

    return div_df, splits_df
