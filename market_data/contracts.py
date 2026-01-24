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


@dataclass
class ContractError:
    field: str
    message: str


@dataclass
class ContractSpec:
    name: str
    required_columns: list[str]
    dtypes: dict[str, str]
    frequency: str
    timezone: str
    keys: list[str]
    allow_missing: dict[str, bool]

    def validate_frame(self, df: pd.DataFrame) -> list[ContractError]:
        errors: list[ContractError] = []
        if df is None or df.empty:
            errors.append(ContractError("frame", "Dataframe is empty."))
            return errors
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            errors.append(ContractError("columns", f"Missing columns: {missing}"))
        for col, dtype in self.dtypes.items():
            if col not in df.columns:
                continue
            series = df[col]
            if dtype == "date" and not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(ContractError(col, "Expected datetime dtype."))
            if dtype == "float" and not pd.api.types.is_numeric_dtype(series):
                errors.append(ContractError(col, "Expected numeric dtype."))
            if dtype == "str" and not pd.api.types.is_string_dtype(series):
                errors.append(ContractError(col, "Expected string dtype."))
        return errors


def PriceSeriesContract() -> ContractSpec:
    return ContractSpec(
        name="PriceSeriesContract",
        required_columns=["date", "close", "adj_close", "dividend", "split_ratio", "ticker", "source"],
        dtypes={
            "date": "date",
            "close": "float",
            "adj_close": "float",
            "dividend": "float",
            "split_ratio": "float",
            "ticker": "str",
            "source": "str",
        },
        frequency="daily",
        timezone="UTC",
        keys=["ticker", "date"],
        allow_missing={"adj_close": False},
    )


def validate_price_series_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise MarketDataError(
            error_code="MARKET_DATA_EMPTY",
            message=f"No market data available for {ticker}.",
            details={"ticker": ticker},
            hint="Ensure price data is available for this ticker.",
        )
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    if "dividend" in frame.columns:
        frame["dividend"] = pd.to_numeric(frame["dividend"], errors="coerce").fillna(0.0)
    if "split_ratio" in frame.columns:
        frame["split_ratio"] = pd.to_numeric(frame["split_ratio"], errors="coerce").fillna(1.0)
    errors = PriceSeriesContract().validate_frame(frame)
    if errors:
        raise MarketDataError(
            error_code="MARKET_DATA_CONTRACT",
            message=f"Price series contract failed for {ticker}.",
            details={"ticker": ticker, "errors": [e.__dict__ for e in errors]},
            hint="Verify market data normalization and required columns.",
        )
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date", "ticker"]).reset_index(drop=True)
    return frame


def RiskFreeSeriesContract() -> ContractSpec:
    return ContractSpec(
        name="RiskFreeSeriesContract",
        required_columns=["date", "rate", "rf_daily_return"],
        dtypes={"date": "date", "rate": "float", "rf_daily_return": "float"},
        frequency="daily",
        timezone="UTC",
        keys=["date"],
        allow_missing={"rf_daily_return": False},
    )


def BenchmarkSeriesContract() -> ContractSpec:
    return ContractSpec(
        name="BenchmarkSeriesContract",
        required_columns=["date", "adj_close", "ticker"],
        dtypes={"date": "date", "adj_close": "float", "ticker": "str"},
        frequency="daily",
        timezone="UTC",
        keys=["ticker", "date"],
        allow_missing={"adj_close": False},
    )


def CoverageSummaryContract(payload: dict[str, Any]) -> list[ContractError]:
    errors: list[ContractError] = []
    for key in [
        "as_of",
        "status",
        "score",
        "policy",
        "required",
        "per_ticker",
        "aggregate",
        "reason_codes",
        "coverage",
        "metric_status",
        "metric_reasons",
    ]:
        if key not in payload:
            errors.append(ContractError(key, "Missing required field."))
    policy = payload.get("policy", {})
    for key in ["min_score_for_kpis", "min_history_days", "max_gap_days"]:
        if key not in policy:
            errors.append(ContractError(f"policy.{key}", "Missing policy field."))
    required = payload.get("required", {})
    for key in ["tickers", "history_days_needed"]:
        if key not in required:
            errors.append(ContractError(f"required.{key}", "Missing required field."))
    aggregate = payload.get("aggregate", {})
    for key in ["coverage_ratio", "min_ticker_score", "benchmark_score", "rf_score"]:
        if key not in aggregate:
            errors.append(ContractError(f"aggregate.{key}", "Missing aggregate field."))
    return errors


def MacroContextContract(payload: dict[str, Any]) -> list[ContractError]:
    errors: list[ContractError] = []
    for key in ["status", "missing_series", "as_of", "flags", "available_series", "tags", "warnings", "cache_status"]:
        if key not in payload:
            errors.append(ContractError(key, "Missing required field."))
    return errors


def EnrichmentContract(payload: dict[str, Any]) -> list[ContractError]:
    errors: list[ContractError] = []
    for key in ["status", "as_of", "provenance", "payload"]:
        if key not in payload:
            errors.append(ContractError(key, "Missing required field."))
    return errors


def contract_registry() -> dict[str, dict[str, str]]:
    return {
        "PriceSeriesContract": {"version": "1.0"},
        "RiskFreeSeriesContract": {"version": "1.0"},
        "BenchmarkSeriesContract": {"version": "1.0"},
        "CoverageSummaryContract": {"version": "2.0"},
        "MacroContextContract": {"version": "1.1"},
        "EnrichmentContract": {"version": "1.0"},
    }


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
