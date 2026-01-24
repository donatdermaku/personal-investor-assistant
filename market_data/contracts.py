from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class MarketDataError(Exception):
    error_code: str
    message: str
    details: dict[str, Any] | None = None
    hint: str | None = None

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


PRICE_REQUIRED = ["date", "close"]


def validate_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise MarketDataError(
            error_code="MARKET_DATA_EMPTY",
            message=f"No market data available for {ticker}.",
            details={"ticker": ticker},
            hint="Ensure price data is available for this ticker.",
        )

    if "date" not in df.columns:
        raise MarketDataError(
            error_code="MARKET_DATA_MISSING_DATE",
            message=f"Market price data missing date column for {ticker}.",
            details={"ticker": ticker, "columns": list(df.columns)},
            hint="Verify the data source returns a date column.",
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if df["date"].isna().any():
        raise MarketDataError(
            error_code="MARKET_DATA_INVALID_DATE",
            message=f"Invalid date values in market data for {ticker}.",
            details={"ticker": ticker},
            hint="Ensure dates are in YYYY-MM-DD format.",
        )

    for col in PRICE_REQUIRED:
        if col not in df.columns:
            raise MarketDataError(
                error_code="MARKET_DATA_MISSING_COLUMNS",
                message=f"Missing required column '{col}' for {ticker}.",
                details={"ticker": ticker, "columns": list(df.columns)},
                hint="Ensure market data includes required fields.",
            )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().any():
        raise MarketDataError(
            error_code="MARKET_DATA_INVALID_CLOSE",
            message=f"Close price is missing or invalid for {ticker}.",
            details={"ticker": ticker},
            hint="Check data source for missing close prices.",
        )

    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def normalize_date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.date

