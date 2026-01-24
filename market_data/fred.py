from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import requests
import pandas as pd

from src.utils_io import ROOT
from market_data.persistent_cache import CacheResult, get_or_refresh_frame, load_cached_frame

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY", "")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(FRED_BASE, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    obs = payload.get("observations", [])
    rows = []
    for item in obs:
        value = item.get("value")
        if value in (".", None):
            continue
        rows.append({"date": item.get("date"), "value": float(value)})
    return pd.DataFrame(rows)


def cache_series(series_id: str, df: pd.DataFrame) -> Path:
    cache_dir = ROOT / "data" / "market_cache" / "fred"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{series_id}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_cached_series(series_id: str) -> pd.DataFrame:
    result = load_cached_frame("fred", series_id)
    return result.frame


def get_cached_series(
    series_id: str,
    *,
    ttl_seconds: int = 86400,
    allow_refresh: bool = False,
    force_refresh: bool = False,
) -> CacheResult:
    return get_or_refresh_frame(
        source="fred",
        key=series_id,
        ttl_seconds=ttl_seconds,
        fetch_fn=lambda: fetch_series(series_id, "1900-01-01", date.today().isoformat()),
        asof_date=date.today().isoformat(),
        allow_refresh=allow_refresh,
        force_refresh=force_refresh,
    )
