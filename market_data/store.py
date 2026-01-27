from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

from market_data.contracts import MarketDataError, validate_price_frame, validate_price_series_frame
from market_data.yahoo import fetch_prices, fetch_dividends_and_splits
from market_data.persistent_cache import get_or_refresh_frame
from src.utils_io import ROOT

# Fixed earliest date for max-history fetching - ensures cache always has full history
FIXED_EARLIEST_DATE = "2010-01-01"


@dataclass
class MarketDataStore:
    cache_dir: Path

    @classmethod
    def default(cls) -> "MarketDataStore":
        base = ROOT / "data" / "market_cache"
        base.mkdir(parents=True, exist_ok=True)
        return cls(cache_dir=base)

    def _prices_path(self, ticker: str) -> Path:
        return self.cache_dir / "prices" / f"{ticker}.parquet"

    def _dividends_path(self, ticker: str) -> Path:
        return self.cache_dir / "dividends" / f"{ticker}.parquet"

    def _splits_path(self, ticker: str) -> Path:
        return self.cache_dir / "splits" / f"{ticker}.parquet"

    def _ensure_dirs(self) -> None:
        (self.cache_dir / "prices").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "dividends").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "splits").mkdir(parents=True, exist_ok=True)

    def _is_price_cache_fresh(
        self, df: pd.DataFrame, required_start: str | None = None
    ) -> bool:
        """Check if cached price data is fresh and covers the required date range.
        
        Args:
            df: Cached price DataFrame
            required_start: Required start date (defaults to FIXED_EARLIEST_DATE)
            
        Returns:
            True if cache covers from required_start to yesterday, False otherwise
        """
        if df.empty or "date" not in df.columns:
            return False
        
        dates = pd.to_datetime(df["date"], errors="coerce")
        first_date = dates.min()
        last_date = dates.max()
        
        if pd.isna(first_date) or pd.isna(last_date):
            return False
        
        # Check end date: must cover at least yesterday
        target_end = (datetime.utcnow().date() - BDay(1)).date()
        if last_date.date() < target_end:
            return False
        
        # Check start date: must cover from required_start (or FIXED_EARLIEST_DATE)
        target_start = pd.to_datetime(required_start or FIXED_EARLIEST_DATE).date()
        if first_date.date() > target_start:
            return False
        
        return True

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker.upper() == "CASH":
            raise MarketDataError(
                error_code="MARKET_DATA_SKIP",
                message="CASH ticker is not eligible for market data.",
                details={"ticker": ticker},
            )
        self._ensure_dirs()
        cache_path = self._prices_path(ticker)
        cached = pd.DataFrame()
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
            except Exception:
                cached = pd.DataFrame()
        
        # Use FIXED_EARLIEST_DATE for fetching to ensure full history coverage
        # Pass the requested start to freshness check for validation
        if cached.empty or not self._is_price_cache_fresh(cached, required_start=start):
            cache_result = get_or_refresh_frame(
                source="yahoo",
                key=ticker,
                ttl_seconds=21600,
                fetch_fn=lambda: fetch_prices(ticker, FIXED_EARLIEST_DATE, end),
                asof_date=end,
                allow_refresh=True,
            )
            cached = cache_result.frame
            
            # Validate before caching to prevent poisoned cache
            if not cached.empty:
                from market_data.rate_limiter import validate_price_cache
                is_valid, reasons = validate_price_cache(
                    cached, 
                    required_start=start,
                    required_end=end,
                    min_rows=1000,  # ~4 years of data; blocks tiny caches but allows IPOs
                )
                if is_valid:
                    cached.to_parquet(cache_path, index=False)
                else:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Cache validation failed for {ticker}, not caching: {reasons}"
                    )
        if not cached.empty and "date" in cached.columns:
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.date
        cached = validate_price_frame(cached, ticker)
        if not cached.empty and "date" in cached.columns:
            date_index = pd.to_datetime(cached["date"], errors="coerce")
            start_date = pd.to_datetime(start, errors="coerce")
            end_date = pd.to_datetime(end, errors="coerce")
            cached = cached[(date_index >= start_date) & (date_index <= end_date)]
        dividends = self.get_dividends(ticker, start, end)
        splits = self.get_splits(ticker, start, end)
        normalized = normalize_price_frame(cached, dividends, splits, source="yahoo")
        return validate_price_series_frame(normalized, ticker)

    def ensure_coverage(self, prices_df: pd.DataFrame, trade_dates: Iterable[date], ticker: str) -> pd.DataFrame:
        if prices_df.empty:
            raise MarketDataError(
                error_code="MARKET_DATA_EMPTY",
                message=f"No market data available for {ticker}.",
                details={"ticker": ticker},
            )
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce").dt.date
        available = set(prices_df["date"].dropna().unique())
        missing: list[str] = []
        aligned: dict[date, date] = {}

        # Helper to ensure date type
        def _to_date(d: Any) -> date | None:
            if isinstance(d, str):
                try:
                    ts = pd.to_datetime(d)
                    return ts.date() if pd.notna(ts) else None
                except Exception:
                    return None
            if isinstance(d, pd.Timestamp):
                return d.date()
            if isinstance(d, datetime):
                return d.date()
            if isinstance(d, date):
                return d
            return None

        # Sort available dates for efficient searching
        sorted_available = sorted(list(available))

        for raw_td in trade_dates:
            td = _to_date(raw_td)
            if td is None:
                # Should we error or skip? Skipping invalid dates seems safest for now
                continue

            if td in available:
                aligned[td] = td
                continue

            # Use previous trading day
            # Find candidate dates strictly less than td
            # Since we sorted_available, we can find the last one < td
            # But simple list comp is fine for usually small list
            prev_days = [d for d in sorted_available if d < td]
            if prev_days:
                aligned[td] = prev_days[-1] # Max of prev_days
            else:
                missing.append(td.isoformat())

        if missing:
            raise MarketDataError(
                error_code="MARKET_DATA_MISSING_DATES",
                message=f"Missing market data dates for {ticker}.",
                details={"ticker": ticker, "missing_dates": missing[:10]},
                hint="Check ticker coverage or adjust trade dates.",
            )
        return prices_df

    def get_dividends(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        self._ensure_dirs()
        path = self._dividends_path(ticker)
        cached = pd.DataFrame()
        if path.exists():
            try:
                cached = pd.read_parquet(path)
            except Exception:
                cached = pd.DataFrame()
        if cached.empty:
            dividends, _ = fetch_dividends_and_splits(ticker, start, end)
            dividends.to_parquet(path, index=False)
            cached = dividends
        return cached

    def get_splits(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        self._ensure_dirs()
        path = self._splits_path(ticker)
        cached = pd.DataFrame()
        if path.exists():
            try:
                cached = pd.read_parquet(path)
            except Exception:
                cached = pd.DataFrame()
        if cached.empty:
            _, splits = fetch_dividends_and_splits(ticker, start, end)
            splits.to_parquet(path, index=False)
            cached = splits
        return cached


def normalize_price_frame(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce").astype("float32")
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce").astype("float32")
    frame["dividend"] = np.float32(0.0)
    frame["split_ratio"] = np.float32(1.0)
    if dividends is not None and not dividends.empty:
        div = dividends.copy()
        div["date"] = pd.to_datetime(div["date"], errors="coerce").dt.date
        div["amount"] = pd.to_numeric(div["amount"], errors="coerce").fillna(0.0)
        div_map = div.groupby("date")["amount"].sum().to_dict()
        frame["dividend"] = frame["date"].map(div_map).fillna(0.0).astype("float32")
    if splits is not None and not splits.empty:
        split = splits.copy()
        split["date"] = pd.to_datetime(split["date"], errors="coerce").dt.date
        split["ratio"] = pd.to_numeric(split["ratio"], errors="coerce").fillna(1.0)
        split_map = split.groupby("date")["ratio"].prod().to_dict()
        frame["split_ratio"] = frame["date"].map(split_map).fillna(1.0).astype("float32")
    frame["source"] = source
    return frame
