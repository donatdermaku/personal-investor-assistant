from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.portfolio import load_portfolio
from src.utils_io import ROOT


DATA_DIR = ROOT / "data"
PARQ_DIR = DATA_DIR / "parquet"


@dataclass
class CoverageMeta:
    total: int
    covered: int
    missing_tickers: list[str]
    last_date: str | None
    reasons: dict[str, int]
    notes: list[str]


def merge_coverage(metas: list[CoverageMeta]) -> CoverageMeta:
    total = sum(meta.total for meta in metas)
    covered = sum(meta.covered for meta in metas)
    missing = sorted({t for meta in metas for t in meta.missing_tickers})
    dates = [meta.last_date for meta in metas if meta.last_date]
    last_date = max(dates) if dates else None
    reasons: dict[str, int] = {}
    notes: list[str] = []
    for meta in metas:
        for key, value in meta.reasons.items():
            reasons[key] = reasons.get(key, 0) + value
        notes.extend(meta.notes)
    return CoverageMeta(total, covered, missing, last_date, reasons, notes)


def _meta_empty(reason: str, note: str | None = None, total: int = 0) -> CoverageMeta:
    notes = [note] if note else []
    return CoverageMeta(total=total, covered=0, missing_tickers=[], last_date=None, reasons={reason: 1}, notes=notes)


def _guess_last_date(df: pd.DataFrame) -> str | None:
    for col in ["date", "asof", "fiscal_end"]:
        if col in df.columns:
            value = pd.to_datetime(df[col], errors="coerce").max()
            if pd.notna(value):
                return value.strftime("%Y-%m-%d")
    return None


def _coverage_from_df(
    df: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
    ticker_col: str = "ticker",
) -> CoverageMeta:
    if df.empty or ticker_col not in df.columns:
        total = len(tickers) if tickers else 0
        missing = tickers or []
        return CoverageMeta(total=total, covered=0, missing_tickers=missing, last_date=_guess_last_date(df), reasons={}, notes=[])
    if tickers:
        present = set(df[ticker_col].dropna().unique())
        missing = [t for t in tickers if t not in present]
        covered = len(tickers) - len(missing)
        total = len(tickers)
    else:
        present = df[ticker_col].dropna().unique()
        total = len(present)
        covered = total
        missing = []
    return CoverageMeta(total=total, covered=covered, missing_tickers=missing, last_date=_guess_last_date(df), reasons={}, notes=[])


def _read_parquet_with_meta(
    path: Path,
    *,
    required_cols: list[str] | None = None,
    tickers: list[str] | None = None,
    ticker_col: str = "ticker",
) -> tuple[pd.DataFrame, CoverageMeta]:
    if not path.exists():
        meta = _meta_empty("missing_file", f"Missing file: {path.name}", total=len(tickers or []))
        return pd.DataFrame(), meta
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        meta = _meta_empty("read_error", f"Read error: {exc}", total=len(tickers or []))
        return pd.DataFrame(), meta
    if required_cols and any(col not in df.columns for col in required_cols):
        meta = _meta_empty("missing_columns", f"Missing columns in {path.name}", total=len(tickers or []))
        return pd.DataFrame(), meta
    if df.empty:
        meta = _meta_empty("empty_data", f"Empty data in {path.name}", total=len(tickers or []))
        return pd.DataFrame(), meta
    return df, _coverage_from_df(df, tickers=tickers, ticker_col=ticker_col)


def _read_csv_with_meta(
    path: Path,
    *,
    required_cols: list[str] | None = None,
    tickers: list[str] | None = None,
    ticker_col: str = "ticker",
) -> tuple[pd.DataFrame, CoverageMeta]:
    if not path.exists():
        meta = _meta_empty("missing_file", f"Missing file: {path.name}", total=len(tickers or []))
        return pd.DataFrame(), meta
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        meta = _meta_empty("read_error", f"Read error: {exc}", total=len(tickers or []))
        return pd.DataFrame(), meta
    if required_cols and any(col not in df.columns for col in required_cols):
        meta = _meta_empty("missing_columns", f"Missing columns in {path.name}", total=len(tickers or []))
        return pd.DataFrame(), meta
    if df.empty:
        meta = _meta_empty("empty_data", f"Empty data in {path.name}", total=len(tickers or []))
        return pd.DataFrame(), meta
    return df, _coverage_from_df(df, tickers=tickers, ticker_col=ticker_col)


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
    return get_scores()[0]


@st.cache_data(ttl=1800)
def load_scores_prior() -> pd.DataFrame:
    return get_scores_prior()[0]


@st.cache_data(ttl=86400)
def load_fundamentals() -> pd.DataFrame:
    return get_fundamentals()[0]


@st.cache_data(ttl=1800)
def load_universe() -> pd.DataFrame:
    return get_universe()[0]


@st.cache_data(ttl=21600)
def _load_prices_open() -> pd.DataFrame:
    return get_prices("open")[0]


@st.cache_data(ttl=21600)
def _load_prices_closed() -> pd.DataFrame:
    return get_prices("closed")[0]


def load_prices(market_state: str) -> pd.DataFrame:
    return get_prices(market_state)[0]


@st.cache_data(ttl=1800)
def load_portfolio_cached(
    prices: pd.DataFrame,
    watchlist: list[str],
    cache_token: tuple[float, float],
    source_override: str | None = None,
    uploads_active: bool = True,
):
    return load_portfolio(prices, watchlist, source_override=source_override, uploads_active=uploads_active)


def portfolio_cache_token() -> tuple[float, float]:
    ledger = DATA_DIR / "user_uploads" / "transactions.csv"
    snapshot = DATA_DIR / "user_uploads" / "holdings.csv"
    ledger_mtime = ledger.stat().st_mtime if ledger.exists() else 0.0
    snapshot_mtime = snapshot.stat().st_mtime if snapshot.exists() else 0.0
    return (ledger_mtime, snapshot_mtime)


@st.cache_data(ttl=21600)
def load_benchmark_prices(ticker: str) -> pd.DataFrame:
    return get_benchmark_prices(ticker)[0]


@st.cache_data(ttl=604800)
def load_news(vendor_ticker: str) -> list[dict]:
    return get_news(vendor_ticker)[0]


@st.cache_data(ttl=1800)
def get_scores(tickers: list[str] | None = None) -> tuple[pd.DataFrame, CoverageMeta]:
    path = _latest_parquet("scores_daily")
    if not path:
        return pd.DataFrame(), _meta_empty("missing_file", "Missing scores_daily parquet", total=len(tickers or []))
    return _read_parquet_with_meta(path, required_cols=["ticker"], tickers=tickers)


@st.cache_data(ttl=1800)
def get_scores_prior(tickers: list[str] | None = None) -> tuple[pd.DataFrame, CoverageMeta]:
    path = _previous_parquet("scores_daily")
    if not path:
        return pd.DataFrame(), _meta_empty("missing_file", "Missing prior scores_daily parquet", total=len(tickers or []))
    return _read_parquet_with_meta(path, required_cols=["ticker"], tickers=tickers)


@st.cache_data(ttl=86400)
def get_fundamentals(tickers: list[str] | None = None) -> tuple[pd.DataFrame, CoverageMeta]:
    path = _latest_parquet("fundamentals_quarterly")
    if not path:
        return pd.DataFrame(), _meta_empty("missing_file", "Missing fundamentals_quarterly parquet", total=len(tickers or []))
    return _read_parquet_with_meta(path, required_cols=["ticker"], tickers=tickers)


@st.cache_data(ttl=1800)
def get_universe() -> tuple[pd.DataFrame, CoverageMeta]:
    path = DATA_DIR / "universe.csv"
    return _read_csv_with_meta(path, required_cols=["ticker"])


@st.cache_data(ttl=21600)
def get_prices(market_state: str, tickers: list[str] | None = None) -> tuple[pd.DataFrame, CoverageMeta]:
    path = _latest_parquet("prices_daily")
    if not path:
        return pd.DataFrame(), _meta_empty("missing_file", "Missing prices_daily parquet", total=len(tickers or []))
    return _read_parquet_with_meta(path, required_cols=["ticker", "adj_close"], tickers=tickers)


@st.cache_data(ttl=21600)
def get_benchmark_prices(ticker: str) -> tuple[pd.DataFrame, CoverageMeta]:
    prices, meta = get_prices("closed")
    if not prices.empty and "ticker" in prices.columns and ticker in prices["ticker"].values:
        bench = prices[prices["ticker"] == ticker].copy()
        return bench, _coverage_from_df(bench, tickers=[ticker])
    return pd.DataFrame(), _meta_empty("missing_benchmark", f"Benchmark data missing for {ticker}", total=1)


@st.cache_data(ttl=604800)
def get_news(vendor_ticker: str) -> tuple[list[dict], CoverageMeta]:
    try:
        import yfinance as yf
    except Exception:
        return [], _meta_empty("missing_dependency", "yfinance not available")
    try:
        ticker = yf.Ticker(vendor_ticker)
        items = ticker.news or []
        meta = CoverageMeta(total=0, covered=0, missing_tickers=[], last_date=None, reasons={}, notes=[])
        if not items:
            meta = _meta_empty("no_news", f"No news for {vendor_ticker}")
        return items, meta
    except Exception as exc:
        return [], _meta_empty("news_error", f"News error: {exc}")


def market_status() -> tuple[str, str]:
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return "Closed", "closed"
    market_open = time(9, 30)
    market_close = time(16, 0)
    if market_open <= now.time() <= market_close:
        return "Open", "open"
    return "Closed", "closed"
