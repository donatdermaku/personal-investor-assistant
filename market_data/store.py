from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.tseries.offsets import BDay

from market_data.contracts import MarketDataError, validate_price_frame, validate_price_series_frame
from market_data.yahoo import fetch_prices, fetch_dividends_and_splits
from src.utils_io import ROOT


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

    def _is_price_cache_fresh(self, df: pd.DataFrame) -> bool:
        if df.empty or "date" not in df.columns:
            return False
        last_date = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(last_date):
            return False
        target = (datetime.utcnow().date() - BDay(1)).date()
        return last_date.date() >= target

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
        if cached.empty or not self._is_price_cache_fresh(cached):
            fetched = fetch_prices(ticker, start, end)
            fetched.to_parquet(cache_path, index=False)
            cached = fetched
        cached = validate_price_frame(cached, ticker)
        cached = cached[(cached["date"] >= date.fromisoformat(start)) & (cached["date"] <= date.fromisoformat(end))]
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
        for td in trade_dates:
            if td in available:
                aligned[td] = td
                continue
            # Use previous trading day
            prev_days = [d for d in available if d < td]
            if prev_days:
                aligned[td] = max(prev_days)
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
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame["dividend"] = 0.0
    frame["split_ratio"] = 1.0
    if dividends is not None and not dividends.empty:
        div = dividends.copy()
        div["date"] = pd.to_datetime(div["date"], errors="coerce").dt.date
        div["amount"] = pd.to_numeric(div["amount"], errors="coerce").fillna(0.0)
        div_map = div.groupby("date")["amount"].sum().to_dict()
        frame["dividend"] = frame["date"].map(div_map).fillna(0.0)
    if splits is not None and not splits.empty:
        split = splits.copy()
        split["date"] = pd.to_datetime(split["date"], errors="coerce").dt.date
        split["ratio"] = pd.to_numeric(split["ratio"], errors="coerce").fillna(1.0)
        split_map = split.groupby("date")["ratio"].prod().to_dict()
        frame["split_ratio"] = frame["date"].map(split_map).fillna(1.0)
    frame["source"] = source
    return frame
