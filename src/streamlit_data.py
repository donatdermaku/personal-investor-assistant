from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.portfolio import load_portfolio
from src.utils_io import ROOT


DATA_DIR = ROOT / "data"
PARQ_DIR = DATA_DIR / "parquet"


def _latest_parquet(prefix: str) -> Path | None:
    files = sorted(PARQ_DIR.glob(f"{prefix}_*.parquet"))
    return files[-1] if files else None


def _previous_parquet(prefix: str) -> Path | None:
    files = sorted(PARQ_DIR.glob(f"{prefix}_*.parquet"))
    if len(files) < 2:
        return None
    return files[-2]


@st.cache_data(ttl=3600)
def load_watchlist() -> dict:
    import yaml

    path = ROOT / "watchlist.yml"
    if not path.exists():
        return {"tickers": []}
    return yaml.safe_load(path.read_text())


@st.cache_data(ttl=1800)
def load_scores() -> pd.DataFrame:
    path = _latest_parquet("scores_daily")
    if not path:
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=1800)
def load_scores_prior() -> pd.DataFrame:
    path = _previous_parquet("scores_daily")
    if not path:
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=86400)
def load_fundamentals() -> pd.DataFrame:
    path = _latest_parquet("fundamentals_quarterly")
    if not path:
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=1800)
def load_universe() -> pd.DataFrame:
    path = DATA_DIR / "universe.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=21600)
def _load_prices_open() -> pd.DataFrame:
    path = _latest_parquet("prices_daily")
    if not path:
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=21600)
def _load_prices_closed() -> pd.DataFrame:
    path = _latest_parquet("prices_daily")
    if not path:
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_prices(market_state: str) -> pd.DataFrame:
    if market_state == "open":
        return _load_prices_open()
    return _load_prices_closed()


@st.cache_data(ttl=1800)
def load_portfolio_cached(prices: pd.DataFrame, watchlist: list[str], cache_token: tuple[float, float]):
    return load_portfolio(prices, watchlist)


def portfolio_cache_token() -> tuple[float, float]:
    ledger = DATA_DIR / "user_uploads" / "transactions.csv"
    snapshot = DATA_DIR / "user_uploads" / "holdings.csv"
    ledger_mtime = ledger.stat().st_mtime if ledger.exists() else 0.0
    snapshot_mtime = snapshot.stat().st_mtime if snapshot.exists() else 0.0
    return (ledger_mtime, snapshot_mtime)


@st.cache_data(ttl=21600)
def load_benchmark_prices(ticker: str) -> pd.DataFrame:
    prices = _load_prices_closed()
    if not prices.empty and ticker in prices["ticker"].values:
        return prices[prices["ticker"] == ticker].copy()
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, start="2015-01-01", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adj_close"})
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        df["ticker"] = ticker
        return df[["date", "ticker", "adj_close"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=604800)
def load_news(vendor_ticker: str) -> list[dict]:
    try:
        import yfinance as yf
    except Exception:
        return []
    try:
        ticker = yf.Ticker(vendor_ticker)
        return ticker.news or []
    except Exception:
        return []


def market_status() -> tuple[str, str]:
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return "Closed", "closed"
    market_open = time(9, 30)
    market_close = time(16, 0)
    if market_open <= now.time() <= market_close:
        return "Open", "open"
    return "Closed", "closed"
