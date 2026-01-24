from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

from src.utils_io import ROOT


def fetch_fundamentals(ticker: str) -> dict:
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info or {}
    return {
        "ticker": ticker,
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "dividendYield": info.get("dividendYield"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


def cache_fundamentals(records: list[dict]) -> Path:
    cache_dir = ROOT / "data" / "market_cache" / "fundamentals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "fundamentals.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    return path

