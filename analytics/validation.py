from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.portfolio import PortfolioResult, align_benchmark


@dataclass
class ValidationResult:
    errors: list[str]

    def ok(self) -> bool:
        return not self.errors


def validate_portfolio_result(result: PortfolioResult) -> ValidationResult:
    errors: list[str] = []

    if not result.daily_values.empty:
        for col in ["value", "cash"]:
            if col in result.daily_values.columns:
                series = result.daily_values[col]
                if np.isinf(series).any() or series.isna().any():
                    errors.append(f"daily_values.{col} contains NaN/inf")

    if not result.daily_returns.empty:
        if np.isinf(result.daily_returns).any() or result.daily_returns.isna().any():
            errors.append("daily_returns contains NaN/inf")

    if result.twr is not None and (np.isnan(result.twr) or np.isinf(result.twr)):
        errors.append("twr is NaN/inf")
    if result.mwr is not None and (np.isnan(result.mwr) or np.isinf(result.mwr)):
        errors.append("mwr is NaN/inf")

    if not result.daily_values.empty:
        drawdown = result.daily_values["value"].cummax()
        drawdown = result.daily_values["value"] / drawdown - 1.0
        if drawdown.dropna().gt(0).any():
            errors.append("drawdown contains positive values")

    return ValidationResult(errors)


def validate_prices_vs_trades(prices: pd.DataFrame, ledger: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    if prices.empty:
        return ValidationResult(["prices empty"])
    if ledger.empty:
        return ValidationResult(errors)

    prices = prices.copy()
    ledger = ledger.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    ledger["date"] = pd.to_datetime(ledger["date"], errors="coerce")

    for ticker in sorted(ledger["ticker"].dropna().unique()):
        if ticker == "CASH":
            continue
        trade_dates = ledger.loc[ledger["ticker"] == ticker, "date"].dropna()
        if trade_dates.empty:
            continue
        price_dates = prices.loc[prices["ticker"] == ticker, "date"].dropna()
        if price_dates.empty:
            errors.append(f"missing price history for {ticker}")
            continue
        min_trade = trade_dates.min()
        max_trade = trade_dates.max()
        if min_trade < price_dates.min():
            errors.append(f"price history starts after first trade for {ticker}")
        if max_trade > price_dates.max():
            errors.append(f"price history ends before last trade for {ticker}")
        missing = trade_dates[~trade_dates.isin(price_dates)]
        if not missing.empty:
            errors.append(f"trade dates missing price rows for {ticker}")

    return ValidationResult(errors)


def validate_benchmark_alignment(benchmark_prices: pd.DataFrame, portfolio_values: pd.Series) -> ValidationResult:
    errors: list[str] = []
    aligned = align_benchmark(benchmark_prices, portfolio_values)
    if aligned.empty or portfolio_values.empty:
        return ValidationResult(errors)
    if not np.isclose(aligned.iloc[0], portfolio_values.iloc[0]):
        errors.append("benchmark alignment does not match portfolio start value")
    if len(aligned) != len(benchmark_prices):
        errors.append("benchmark alignment length mismatch")
    return ValidationResult(errors)


def validate_risk_metrics(metrics: dict[str, float | None]) -> ValidationResult:
    errors: list[str] = []
    var_95 = metrics.get("var_95")
    cvar_95 = metrics.get("cvar_95")
    volatility = metrics.get("volatility")
    sharpe = metrics.get("sharpe")

    if var_95 is not None and cvar_95 is not None and cvar_95 > var_95:
        errors.append("cvar_95 exceeds var_95")
    if volatility is not None and volatility < 0:
        errors.append("volatility is negative")
    if volatility in (0, 0.0) and sharpe not in (None, 0, 0.0):
        errors.append("sharpe should be None when volatility is zero")
    return ValidationResult(errors)


def validate_regression_hash(series: Iterable[tuple[str, float]], expected_hash: str) -> ValidationResult:
    import hashlib

    payload = "|".join(f"{d}:{v:.10f}" for d, v in series)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if digest != expected_hash:
        return ValidationResult([f"hash mismatch: {digest} != {expected_hash}"])
    return ValidationResult([])
