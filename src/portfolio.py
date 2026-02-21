from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


from storage.datamanager import data_manager

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


def compute_twr(
    valuation_series: pd.Series,
    external_cashflows: pd.Series | None,
) -> tuple[float | None, pd.Series]:
    if valuation_series.empty:
        return None, pd.Series(dtype=float)
    if external_cashflows is None or external_cashflows.empty:
        cf = pd.Series(0.0, index=valuation_series.index)
    else:
        cf = external_cashflows.reindex(valuation_series.index).fillna(0.0)
    prev = valuation_series.shift(1)
    raw_daily = (valuation_series - cf) / prev - 1.0
    valid = prev > 0
    daily = raw_daily.where(valid)

    # Start a new linked sub-period after a zero-balance day that receives a flow.
    relink_points = (prev == 0) & (cf != 0)
    subperiod_id = relink_points.cumsum()

    linked_terms: list[float] = []
    for _, group in daily.groupby(subperiod_id):
        valid_group = pd.to_numeric(group, errors="coerce").dropna()
        if valid_group.empty:
            continue
        linked_terms.append(float((1.0 + valid_group).prod() - 1.0))

    if not linked_terms:
        return None, daily
    twr = float(np.prod([1.0 + term for term in linked_terms]) - 1.0)
    return twr, daily


def compute_irr(
    cashflows: pd.Series,
    terminal_value: float | None,
    valuation_end_date: pd.Timestamp | str | None = None,
) -> float | None:
    if valuation_end_date is None:
        raise ValueError("valuation_end_date is required for IRR calculation.")
    return _xirr(cashflows, terminal_value, valuation_end_date)


def compute_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    if daily_returns.empty:
        return pd.Series(dtype=float)
    return daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def compute_drawdown(valuation_series: pd.Series) -> pd.Series:
    if valuation_series.empty:
        return pd.Series(dtype=float)
    peak = valuation_series.cummax()
    return valuation_series / peak - 1.0


def align_benchmark(benchmark_prices: pd.DataFrame, portfolio_values: pd.Series) -> pd.Series:
    if benchmark_prices.empty or portfolio_values.empty:
        return pd.Series(dtype=float)
    bench = benchmark_prices.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.sort_values("date")
    if bench["adj_close"].empty:
        return pd.Series(dtype=float)
    
    # Protection against division by zero or NaN
    first_price = bench["adj_close"].iloc[0]
    if pd.isna(first_price) or first_price == 0:
        return pd.Series(dtype=float)  # Return empty series instead of crashing
    
    scaled = (bench["adj_close"] / first_price) * portfolio_values.iloc[0]
    return pd.Series(scaled.values, index=bench["date"])


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


def load_portfolio(
    prices: pd.DataFrame,
    watchlist: Iterable[str],
    source_override: str | None = None,
    uploads_active: bool = True,
    base_dir: Path | None = None,
) -> PortfolioResult:
    # Resolve Portfolio ID (assuming default user/portfolio for now)
    user_id = data_manager.get_current_user_id()
    portfolio_id = data_manager.get_main_portfolio_id(user_id)

    if not uploads_active and source_override != "Demo":
        return PortfolioResult("none", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No uploads in this session."])

    # Load Inputs
    ledger_df = data_manager.load_trades(portfolio_id)
    snapshot_df = data_manager.load_snapshot(portfolio_id)

    if source_override == "Ledger":
        if ledger_df.empty:
            return PortfolioResult("ledger", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["Ledger data empty."])
        return compute_portfolio_from_ledger(ledger_df, prices)

    if source_override == "Snapshot":
        if snapshot_df.empty:
            return PortfolioResult("snapshot", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["Snapshot data empty."])
        return compute_portfolio_from_snapshot(snapshot_df, prices)

    if source_override == "Demo":
        demo = demo_portfolio(watchlist, prices)
        if demo.empty:
            return PortfolioResult("demo", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No demo portfolio available."])
        return compute_portfolio_from_snapshot(demo, prices)

    # Auto-detect
    if not ledger_df.empty:
        result = compute_portfolio_from_ledger(ledger_df, prices)
        if not result.errors:
            return result

    if not snapshot_df.empty:
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

    if "date" not in prices.columns:
        return PortfolioResult(
            "ledger",
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.Series(dtype=float),
            None,
            None,
            ["Market price data missing date column."],
        )

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_dates = prices["date"].dropna().sort_values().unique()
    if len(price_dates) == 0:
        return PortfolioResult("ledger", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No price data available."])

    # Only compute from the first trade date onward — avoids iterating over
    # years of irrelevant history (e.g. 2010→2024) that produce zero-value rows
    first_trade = ledger["date"].min()
    if pd.notna(first_trade):
        price_dates = price_dates[price_dates >= first_trade]
    if len(price_dates) == 0:
        return PortfolioResult("ledger", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, ["No price data after first trade date."])

    holdings = {}
    cash = 0.0
    daily_values = []
    holdings_snapshots = []  # Track holdings state at each date
    cashflows = []
    cashflows_mwr = []

    # Group ledger entries by trading day, snapping non-trading days
    # (holidays, weekends) to the NEXT available trading day so that
    # deposits/trades on e.g. MLK Day or Labor Day are not silently dropped.
    price_dates_set = set(price_dates)
    ledger_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for ledger_date, grp in ledger.groupby(ledger["date"].dt.normalize()):
        if ledger_date in price_dates_set:
            target = ledger_date
        else:
            # Find next available trading day
            future = [d for d in price_dates if d > ledger_date]
            if future:
                target = pd.Timestamp(future[0]).normalize()
            else:
                # No future trading day — snap to last available
                target = pd.Timestamp(price_dates[-1]).normalize()
        if target in ledger_by_date:
            ledger_by_date[target] = pd.concat([ledger_by_date[target], grp])
        else:
            ledger_by_date[target] = grp

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
                if action in {"DEPOSIT", "WITHDRAWAL"}:
                    mwr_flow = -abs(price) if action == "DEPOSIT" else abs(price)
                    cashflows_mwr.append((day, mwr_flow))

        snapshot = prices[prices["date"] == d]
        value = cash
        for ticker, qty in holdings.items():
            price_row = snapshot[snapshot["ticker"] == ticker]
            if not price_row.empty:
                value += qty * float(price_row.iloc[0]["adj_close"])
        daily_values.append({"date": d, "value": value, "cash": cash})
        # Snapshot the current holdings state (not just the final state)
        for ticker, qty in holdings.items():
            if qty != 0:
                holdings_snapshots.append({"date": d, "ticker": ticker, "quantity": qty})

    values_df = pd.DataFrame(daily_values).set_index("date")
    if cashflows:
        cashflows_series = pd.DataFrame(cashflows, columns=["date", "amount"]).groupby("date")["amount"].sum()
    else:
        cashflows_series = pd.Series(dtype=float)
    if cashflows_mwr:
        cashflows_mwr_series = pd.DataFrame(cashflows_mwr, columns=["date", "amount"]).groupby("date")["amount"].sum()
    else:
        cashflows_mwr_series = pd.Series(dtype=float)

    twr, daily_returns = compute_twr(values_df["value"], cashflows_series)
    mwr = compute_irr(
        cashflows_mwr_series,
        values_df["value"].iloc[-1] if not values_df.empty else None,
        valuation_end_date=values_df.index[-1] if not values_df.empty else None,
    )

    returns = daily_returns.fillna(0.0) if not daily_returns.empty else values_df["value"].pct_change().fillna(0.0)
    holdings_daily = pd.DataFrame(holdings_snapshots) if holdings_snapshots else pd.DataFrame(columns=["date", "ticker", "quantity"])

    return PortfolioResult("ledger", values_df, returns, holdings_daily, cashflows_series, mwr, twr, errors)


def compute_portfolio_from_snapshot(snapshot: pd.DataFrame, prices: pd.DataFrame) -> PortfolioResult:
    snapshot, errors = load_snapshot(snapshot)
    if errors:
        return PortfolioResult("snapshot", pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), None, None, errors)

    if "date" not in prices.columns:
        return PortfolioResult(
            "snapshot",
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.Series(dtype=float),
            None,
            None,
            ["Market price data missing date column."],
        )

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
    twr, returns = compute_twr(values_df["value"], None)
    return PortfolioResult("snapshot", values_df, returns, snapshot.reset_index(), pd.Series(dtype=float), None, twr, errors)


def _expand_holdings(holdings: dict[str, float], prices: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    records = []
    for d in dates:
        for ticker, qty in holdings.items():
            records.append({"date": d, "ticker": ticker, "quantity": qty})
    return pd.DataFrame(records)


def _xirr(
    cashflows: pd.Series,
    terminal_value: float | None,
    valuation_end_date: pd.Timestamp | str,
) -> float | None:
    if terminal_value is None or cashflows.empty:
        return None
    dates = [pd.Timestamp(d) for d in cashflows.index]
    amounts = list(cashflows.values)
    dates.append(pd.Timestamp(valuation_end_date))
    amounts.append(terminal_value)

    def npv(rate: float) -> float:
        if rate <= -0.999999:
            return np.nan
        total = 0.0
        for d, cf in zip(dates, amounts):
            days = (d - dates[0]).days
            total += cf / (1 + rate) ** (days / 365)
        return total

    rate = 0.1
    for _ in range(100):
        f = npv(rate)
        if f != f:
            break
        if abs(f) < 1e-6:
            return rate
        # derivative approximation
        f1 = npv(rate + 1e-5)
        if f1 != f1:
            break
        derivative = (f1 - f) / 1e-5
        if derivative == 0:
            break
        rate -= f / derivative
        if rate <= -0.999999:
            break

    low, high = -0.9, 10.0
    f_low = npv(low)
    f_high = npv(high)
    if f_low != f_low or f_high != f_high or f_low * f_high > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        f_mid = npv(mid)
        if f_mid != f_mid:
            return None
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2
