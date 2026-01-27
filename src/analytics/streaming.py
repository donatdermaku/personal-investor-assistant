from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from market_data.calendar import canonical_market_calendar


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def build_canonical_calendar(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
    total_values: pd.DataFrame | pd.Series | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DatetimeIndex:
    """Compute a canonical market calendar spanning available inputs."""
    candidates: list[date] = []
    if start is not None:
        candidates.append(start)
    if end is not None:
        candidates.append(end)

    if prices is not None and not prices.empty and "date" in prices.columns:
        min_date = _coerce_date(prices["date"].min())
        max_date = _coerce_date(prices["date"].max())
        if min_date:
            candidates.append(min_date)
        if max_date:
            candidates.append(max_date)

    if benchmark_prices is not None and not benchmark_prices.empty and "date" in benchmark_prices.columns:
        min_date = _coerce_date(benchmark_prices["date"].min())
        max_date = _coerce_date(benchmark_prices["date"].max())
        if min_date:
            candidates.append(min_date)
        if max_date:
            candidates.append(max_date)

    if total_values is not None and not isinstance(total_values, pd.DataFrame) and hasattr(total_values, "index"):
        if len(total_values.index):
            min_date = _coerce_date(total_values.index.min())
            max_date = _coerce_date(total_values.index.max())
            if min_date:
                candidates.append(min_date)
            if max_date:
                candidates.append(max_date)

    if total_values is not None and isinstance(total_values, pd.DataFrame) and not total_values.empty:
        if "date" in total_values.columns:
            min_date = _coerce_date(total_values["date"].min())
            max_date = _coerce_date(total_values["date"].max())
        else:
            min_date = _coerce_date(total_values.index.min())
            max_date = _coerce_date(total_values.index.max())
        if min_date:
            candidates.append(min_date)
        if max_date:
            candidates.append(max_date)

    if not candidates:
        return pd.DatetimeIndex([])

    start_date = min(candidates) if start is None else start
    end_date = max(candidates) if end is None else end
    if start_date is None or end_date is None:
        return pd.DatetimeIndex([])

    calendar, _ = canonical_market_calendar(
        start=start_date,
        end=end_date,
        benchmark_prices=benchmark_prices,
        ticker_prices=prices,
    )
    return calendar


def iter_price_state(
    prices: pd.DataFrame,
    tickers: list[str],
    calendar: pd.DatetimeIndex,
) -> Iterator[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    """Yield (date, current_price, returns) for each calendar date."""
    if prices is None or prices.empty or not tickers or calendar.empty:
        return

    base = prices.loc[:, ["date", "ticker", "adj_close"]]
    base = base.dropna(subset=["date", "ticker", "adj_close"])
    base = base[base["ticker"].isin(tickers)]
    if base.empty:
        return

    base = base.assign(
        date=pd.to_datetime(base["date"], errors="coerce").dt.normalize(),
        adj_close=pd.to_numeric(base["adj_close"], errors="coerce").astype("float32"),
    ).dropna(subset=["date", "ticker", "adj_close"])

    if base.empty:
        return

    base = base.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    grouped = iter(base.groupby("date", sort=True))
    next_price_date, price_day = next(grouped, (None, None))

    ticker_index = {ticker: idx for idx, ticker in enumerate(tickers)}
    n_assets = len(tickers)
    last_price = np.full(n_assets, np.nan, dtype=np.float32)

    for date in calendar:
        prev_price = last_price.copy()
        if next_price_date is not None:
            while next_price_date is not None and next_price_date <= date:
                for row in price_day.itertuples(index=False):
                    idx = ticker_index.get(row.ticker)
                    if idx is not None:
                        last_price[idx] = float(row.adj_close)
                next_price_date, price_day = next(grouped, (None, None))
                if next_price_date is None or next_price_date > date:
                    break

        current_price = last_price.copy()
        returns = np.zeros(n_assets, dtype=np.float32)
        valid = (~np.isnan(prev_price)) & (~np.isnan(current_price))
        returns[valid] = (current_price[valid] / prev_price[valid]) - np.float32(1.0)
        yield pd.Timestamp(date), current_price, returns


def iter_portfolio_state(
    prices: pd.DataFrame,
    holdings_daily: pd.DataFrame,
    total_values: pd.Series | pd.DataFrame,
    tickers: list[str],
    calendar: pd.DatetimeIndex,
) -> Iterator[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    """Yield (date, returns, weights) aligned to calendar."""
    if holdings_daily is None or holdings_daily.empty or not tickers or calendar.empty:
        return

    holdings = holdings_daily.loc[:, ["date", "ticker", "quantity"]]
    holdings = holdings.dropna(subset=["date", "ticker"])
    holdings = holdings[holdings["ticker"].isin(tickers)]
    if holdings.empty:
        return

    holdings = holdings.assign(
        date=pd.to_datetime(holdings["date"], errors="coerce").dt.normalize(),
        quantity=pd.to_numeric(holdings["quantity"], errors="coerce").fillna(0.0).astype("float32"),
    ).dropna(subset=["date", "ticker"])
    holdings = holdings.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    holdings_grouped = iter(holdings.groupby("date", sort=True))
    next_hold_date, hold_day = next(holdings_grouped, (None, None))

    if isinstance(total_values, pd.DataFrame):
        if "value" in total_values.columns:
            values_series = total_values["value"]
        else:
            values_series = total_values.iloc[:, 0]
    else:
        values_series = total_values
    values_series = values_series.copy() if values_series is not None else pd.Series(dtype=float)
    values_series.index = pd.to_datetime(values_series.index, errors="coerce").normalize()
    aligned_values = values_series.reindex(calendar).fillna(0.0).astype("float32").to_numpy()

    ticker_index = {ticker: idx for idx, ticker in enumerate(tickers)}
    n_assets = len(tickers)
    current_qty = np.zeros(n_assets, dtype=np.float32)

    price_iter = iter_price_state(prices, tickers, calendar)
    for idx, (date, current_price, returns) in enumerate(price_iter):
        if next_hold_date is not None:
            while next_hold_date is not None and next_hold_date <= date:
                for row in hold_day.itertuples(index=False):
                    t_idx = ticker_index.get(row.ticker)
                    if t_idx is not None:
                        current_qty[t_idx] = float(row.quantity)
                next_hold_date, hold_day = next(holdings_grouped, (None, None))
                if next_hold_date is None or next_hold_date > date:
                    break

        safe_price = np.nan_to_num(current_price, nan=0.0)
        holdings_value = current_qty * safe_price
        total_value = aligned_values[idx] if idx < len(aligned_values) else 0.0
        if total_value > 0:
            weights = holdings_value / total_value
        else:
            weights = np.zeros(n_assets, dtype=np.float32)
        yield date, returns, weights


@dataclass
class OnlineCovariance:
    n_assets: int
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.n_assets, dtype=np.float64)
        self.m2 = np.zeros((self.n_assets, self.n_assets), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x is None or len(x) != self.n_assets:
            return
        self.count += 1
        x = x.astype(np.float64, copy=False)
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += np.outer(delta, delta2)

    def covariance(self) -> np.ndarray | None:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)
