from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import logging
import pandas as pd
import numpy as np

from market_data.contracts import MarketDataError, validate_price_frame, validate_price_series_frame
from market_data.yahoo import fetch_prices, fetch_dividends_and_splits
from market_data.persistent_cache import get_or_refresh_frame
from src.utils_io import ROOT

logger = logging.getLogger(__name__)

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

    # ── Cache TTL constants ─────────────────────────
    PRICE_TTL = 21_600       # 6 hours
    DIVIDEND_TTL = 604_800   # 7 days
    SPLIT_TTL = 604_800      # 7 days

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch price data via the unified persistent cache (single layer).

        Cache freshness is controlled entirely by ``PRICE_TTL`` inside
        ``get_or_refresh_frame`` — no secondary date-range check.
        """
        if ticker.upper() == "CASH":
            raise MarketDataError(
                error_code="MARKET_DATA_SKIP",
                message="CASH ticker is not eligible for market data.",
                details={"ticker": ticker},
            )

        # ── Single-layer fetch via persistent cache ──────────
        try:
            cache_result = get_or_refresh_frame(
                source="yahoo",
                key=ticker,
                ttl_seconds=self.PRICE_TTL,
                fetch_fn=lambda: fetch_prices(ticker, FIXED_EARLIEST_DATE, end),
                asof_date=end,
                allow_refresh=True,
            )
            cached = cache_result.frame
            logger.debug(
                "CACHE_HIT ticker=%s status=%s", ticker, cache_result.status,
            )
        except MarketDataError:
            raise
        except Exception as exc:
            logger.error("CACHE_FETCH_FAILED ticker=%s error=%s", ticker, exc)
            raise MarketDataError(
                error_code="MARKET_DATA_FETCH_FAILED",
                message=f"Failed to fetch market data for {ticker}.",
                details={"ticker": ticker, "error": str(exc)},
            )

        # ── Post-fetch validation ────────────────────────────
        if cached.empty:
            raise MarketDataError(
                error_code="MARKET_DATA_FETCH_EMPTY",
                message=f"Yahoo Finance returned no data for {ticker}",
                details={"ticker": ticker, "start": FIXED_EARLIEST_DATE, "end": end},
                hint="Verify ticker symbol is valid and has trading history.",
            )

        if "date" not in cached.columns:
            raise MarketDataError(
                error_code="MARKET_DATA_MALFORMED",
                message=f"Fetched data for {ticker} is missing date column",
                details={
                    "ticker": ticker,
                    "columns_received": list(cached.columns),
                    "shape": cached.shape,
                },
                hint="Yahoo Finance API may have changed. Check logs.",
            )

        # Validate data quality (but don't gate caching — TTL handles that)
        from market_data.rate_limiter import validate_price_cache
        is_valid, reasons = validate_price_cache(
            cached, required_start=start, required_end=end, min_rows=50,
        )
        if not is_valid:
            logger.warning("Price validation warnings for %s: %s", ticker, reasons)
            # If critical (end not covered), clear and retry once
            if any("END_NOT_COVERED" in r for r in reasons):
                logger.warning("Stale cache for %s — clearing + retrying", ticker)
                from market_data.persistent_cache import clear_stale_cache
                clear_stale_cache(source="yahoo", key=ticker)
                try:
                    retry_result = get_or_refresh_frame(
                        source="yahoo",
                        key=ticker,
                        ttl_seconds=self.PRICE_TTL,
                        fetch_fn=lambda: fetch_prices(ticker, FIXED_EARLIEST_DATE, end),
                        asof_date=end,
                        allow_refresh=True,
                    )
                    cached = retry_result.frame
                except Exception as retry_exc:
                    raise MarketDataError(
                        error_code="MARKET_DATA_STALE",
                        message=f"Market data for {ticker} stale after refresh.",
                        details={"ticker": ticker, "error": str(retry_exc)},
                        hint="Clear cache manually or check Yahoo Finance.",
                    )

        # ── Date filtering ───────────────────────────────────
        if not cached.empty and "date" in cached.columns:
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.date
        cached = validate_price_frame(cached, ticker)
        if not cached.empty and "date" in cached.columns:
            date_index = pd.to_datetime(cached["date"], errors="coerce")
            start_date = pd.to_datetime(start, errors="coerce")
            end_date = pd.to_datetime(end, errors="coerce")
            cached = cached[(date_index >= start_date) & (date_index <= end_date)]

        # ── Merge dividends/splits and normalize ─────────────
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
        """Fetch dividends via persistent cache with 7-day TTL."""
        try:
            result = get_or_refresh_frame(
                source="yahoo_dividends",
                key=ticker,
                ttl_seconds=self.DIVIDEND_TTL,
                fetch_fn=lambda: fetch_dividends_and_splits(ticker, start, end)[0],
                asof_date=end,
                allow_refresh=True,
            )
            return result.frame
        except Exception as exc:
            logger.warning("Dividend fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def get_splits(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch splits via persistent cache with 7-day TTL."""
        try:
            result = get_or_refresh_frame(
                source="yahoo_splits",
                key=ticker,
                ttl_seconds=self.SPLIT_TTL,
                fetch_fn=lambda: fetch_dividends_and_splits(ticker, start, end)[1],
                asof_date=end,
                allow_refresh=True,
            )
            return result.frame
        except Exception as exc:
            logger.warning("Split fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()


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
