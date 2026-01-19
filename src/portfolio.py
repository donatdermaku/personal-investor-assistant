from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


LEDGER_REQUIRED = ["date", "ticker", "action", "quantity", "price"]
SNAPSHOT_REQUIRED = ["ticker", "quantity"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: str(c).strip().lower() for c in df.columns}
    return df.rename(columns=cols)


def _coerce_action(value: str) -> str:
    raw = str(value or "").strip().upper()
    return raw


def validate_ledger(df: pd.DataFrame, allow_short: bool = False) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    df = _normalize_columns(df)
    for col in LEDGER_REQUIRED:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    if errors:
        return pd.DataFrame(), errors

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].apply(_coerce_action)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "fees" in df.columns:
        df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)
    else:
        df["fees"] = 0.0

    for idx, row in df.iterrows():
        action = row["action"]
        ticker = row["ticker"]
        if pd.isna(row["date"]):
            errors.append(f"Row {idx}: invalid date for {ticker}")
            continue
        if action in {"BUY", "SELL"}:
            if ticker != "CASH":
                if row["quantity"] is None or row["quantity"] <= 0:
                    errors.append(f"Row {idx}: {action} requires quantity > 0 for {ticker}")
            if row["price"] is None or row["price"] <= 0:
                errors.append(f"Row {idx}: {action} requires price > 0 for {ticker}")
        elif action == "DIVIDEND":
            qty = row.get("quantity")
            price = row.get("price")
            if (pd.isna(qty) or qty == 0) and (pd.isna(price) or price == 0):
                errors.append(f"Row {idx}: DIVIDEND requires quantity*price or price for {ticker}")
        elif action in {"DEPOSIT", "WITHDRAWAL", "FEE", "INTEREST"}:
            if row["price"] is None or row["price"] == 0:
                errors.append(f"Row {idx}: {action} requires price (cash amount)")
        else:
            errors.append(f"Row {idx}: unknown action {action} for {ticker}")

    if errors:
        return pd.DataFrame(), errors

    if not allow_short:
        holdings = {}
        for idx, row in df.sort_values("date").iterrows():
            action = row["action"]
            ticker = row["ticker"]
            if action == "BUY":
                holdings[ticker] = holdings.get(ticker, 0.0) + row["quantity"]
            elif action == "SELL":
                holdings[ticker] = holdings.get(ticker, 0.0) - row["quantity"]
                if holdings[ticker] < 0:
                    errors.append(f"Row {idx}: SELL creates negative holdings for {ticker}")
    if errors:
        return pd.DataFrame(), errors

    return df.sort_values("date"), errors


def load_snapshot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    df = _normalize_columns(df)
    for col in SNAPSHOT_REQUIRED:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    if errors:
        return pd.DataFrame(), errors
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out = out.dropna(subset=["ticker", "quantity"])
    return out, errors


def demo_portfolio(tickers: Iterable[str], prices: pd.DataFrame) -> pd.DataFrame:
    tickers = list(tickers)
    if not tickers or prices.empty:
        return pd.DataFrame(columns=["ticker", "quantity"])
    latest = prices[prices["ticker"].isin(tickers)].copy()
    latest = latest.sort_values("date").groupby("ticker").tail(1)
    budget = 100000.0
    per = budget / len(tickers)
    latest["quantity"] = per / latest["adj_close"].replace({0: np.nan})
    return latest[["ticker", "quantity"]].dropna()


@dataclass
class PortfolioResult:
    source: str
    daily_values: pd.DataFrame
    daily_returns: pd.Series
    holdings_daily: pd.DataFrame
    cashflows: pd.Series
    mwr: float | None
    twr: float | None
    errors: list[str]


def load_portfolio(prices: pd.DataFrame, watchlist: Iterable[str]) -> PortfolioResult:
    base_dir = Path(__file__).resolve().parents[1] / "data" / "user_uploads"
    ledger_path = base_dir / "transactions.csv"
    snapshot_path = base_dir / "holdings.csv"

    if ledger_path.exists():
        ledger_df = pd.read_csv(ledger_path)
        result = compute_portfolio_from_ledger(ledger_df, prices)
        if not result.errors:
            return result

    if snapshot_path.exists():
        snapshot_df = pd.read_csv(snapshot_path)
        result = compute_portfolio_from_snapshot(snapshot_df, prices)
        if not result.errors:
            return result

    demo = demo_portfolio(watchlist, prices)
    if demo.empty:
        return PortfolioResult("demo", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No demo portfolio available."])
    return compute_portfolio_from_snapshot(demo, prices)


def compute_portfolio_from_ledger(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    allow_short: bool = False,
) -> PortfolioResult:
    ledger, errors = validate_ledger(ledger, allow_short=allow_short)
    if errors:
        return PortfolioResult("ledger", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, errors)

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_dates = prices["date"].dropna().sort_values().unique()
    if len(price_dates) == 0:
        return PortfolioResult("ledger", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No price data available."])

    holdings = {}
    cash = 0.0
    daily_values = []
    cashflows = []
    cashflows_mwr = []

    ledger_by_date = {d: g for d, g in ledger.groupby(ledger["date"].dt.normalize())}

    for d in price_dates:
        day = pd.Timestamp(d).normalize()
        if day in ledger_by_date:
            for _, row in ledger_by_date[day].iterrows():
                action = row["action"]
                ticker = row["ticker"]
                qty = row.get("quantity") or 0.0
                price = row.get("price") or 0.0
                fees = row.get("fees") or 0.0
                if action == "BUY":
                    if ticker == "CASH":
                        cash += price
                    else:
                        holdings[ticker] = holdings.get(ticker, 0.0) + qty
                        cash -= qty * price + fees
                elif action == "SELL":
                    if ticker == "CASH":
                        cash -= price
                    else:
                        holdings[ticker] = holdings.get(ticker, 0.0) - qty
                        cash += qty * price - fees
                elif action == "DIVIDEND":
                    if qty and price:
                        cash += qty * price
                    else:
                        cash += price
                elif action == "DEPOSIT":
                    cash += price
                elif action == "WITHDRAWAL":
                    cash -= price
                elif action == "FEE":
                    cash -= price
                elif action == "INTEREST":
                    cash += price

                if action in {"DEPOSIT", "WITHDRAWAL"}:
                    flow = price if action != "WITHDRAWAL" else -abs(price)
                    cashflows.append((day, flow))
                if action in {"DEPOSIT", "WITHDRAWAL", "DIVIDEND", "FEE", "INTEREST"}:
                    flow = price if action not in {"WITHDRAWAL", "FEE"} else -abs(price)
                    mwr_flow = -abs(price) if action == "DEPOSIT" else flow
                    cashflows_mwr.append((day, mwr_flow))

        snapshot = prices[prices["date"] == d]
        value = cash
        for ticker, qty in holdings.items():
            price_row = snapshot[snapshot["ticker"] == ticker]
            if not price_row.empty:
                value += qty * float(price_row.iloc[0]["adj_close"])
        daily_values.append({"date": d, "value": value, "cash": cash})

    values_df = pd.DataFrame(daily_values).set_index("date")
    if cashflows:
        cashflows_series = pd.DataFrame(cashflows, columns=["date", "amount"]).groupby("date")["amount"].sum()
    else:
        cashflows_series = pd.Series(dtype=float)
    if cashflows_mwr:
        cashflows_mwr_series = pd.DataFrame(cashflows_mwr, columns=["date", "amount"]).groupby("date")["amount"].sum()
    else:
        cashflows_mwr_series = pd.Series(dtype=float)

    twr = None
    daily_returns = pd.Series(dtype=float)
    if not values_df.empty:
        cf = cashflows_series.reindex(values_df.index).fillna(0.0)
        prev = values_df["value"].shift(1)
        daily = (values_df["value"] - cf) / prev - 1.0
        daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        twr = (1 + daily).prod() - 1
        daily_returns = daily
    mwr = _xirr(cashflows_mwr_series, values_df["value"].iloc[-1] if not values_df.empty else None)

    returns = daily_returns if not daily_returns.empty else values_df["value"].pct_change().fillna(0.0)
    holdings_daily = _expand_holdings(holdings, prices, values_df.index)

    return PortfolioResult("ledger", values_df, returns, holdings_daily, cashflows_series, mwr, twr, errors)


def compute_portfolio_from_snapshot(snapshot: pd.DataFrame, prices: pd.DataFrame) -> PortfolioResult:
    snapshot, errors = load_snapshot(snapshot)
    if errors:
        return PortfolioResult("snapshot", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, errors)

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_dates = prices["date"].dropna().sort_values().unique()
    if len(price_dates) == 0:
        return PortfolioResult("snapshot", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No price data available."])

    snapshot = snapshot.set_index("ticker")
    daily_values = []
    for d in price_dates:
        day_prices = prices[prices["date"] == d].set_index("ticker")
        value = 0.0
        for ticker, row in snapshot.iterrows():
            if ticker in day_prices.index:
                value += row["quantity"] * day_prices.loc[ticker, "adj_close"]
        daily_values.append({"date": d, "value": value, "cash": 0.0})

    values_df = pd.DataFrame(daily_values).set_index("date")
    returns = values_df["value"].pct_change().fillna(0.0)
    twr = (1 + returns).prod() - 1 if not values_df.empty else None
    return PortfolioResult("snapshot", values_df, returns, snapshot.reset_index(), pd.Series(dtype=float), None, twr, errors)


def _expand_holdings(holdings: dict[str, float], prices: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    records = []
    for d in dates:
        for ticker, qty in holdings.items():
            records.append({"date": d, "ticker": ticker, "quantity": qty})
    return pd.DataFrame(records)


def _xirr(cashflows: pd.Series, terminal_value: float | None) -> float | None:
    if terminal_value is None or cashflows.empty:
        return None
    dates = list(cashflows.index)
    amounts = list(cashflows.values)
    dates.append(dates[-1])
    amounts.append(terminal_value)

    def npv(rate: float) -> float:
        total = 0.0
        for d, cf in zip(dates, amounts):
            days = (d - dates[0]).days
            total += cf / (1 + rate) ** (days / 365)
        return total

    rate = 0.1
    for _ in range(100):
        f = npv(rate)
        if abs(f) < 1e-6:
            return rate
        # derivative approximation
        f1 = npv(rate + 1e-5)
        derivative = (f1 - f) / 1e-5
        if derivative == 0:
            break
        rate -= f / derivative
    return None
