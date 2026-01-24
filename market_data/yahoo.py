from __future__ import annotations

from datetime import date
from typing import Tuple

import pandas as pd
import yfinance as yf

from market_data.contracts import MarketDataError, validate_price_frame


def _normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    return df


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
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
    try:
        yf_ticker = yf.Ticker(ticker)
        dividends = yf_ticker.dividends
        splits = yf_ticker.splits
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

