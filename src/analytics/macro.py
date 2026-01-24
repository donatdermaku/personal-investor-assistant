from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils_io import ROOT


def load_cached_fred_series(series_id: str) -> pd.DataFrame:
    cache_dir = ROOT / "data" / "market_cache" / "fred"
    path = cache_dir / f"{series_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    return df.sort_values("date")


def compute_macro_regime_flags(
    dates: pd.DatetimeIndex,
    cpi: pd.DataFrame,
    fed_funds: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    if dates.empty:
        return pd.DataFrame()

    base = pd.DataFrame({"date": pd.to_datetime(dates)}).dropna().drop_duplicates().sort_values("date")

    out = base.copy()
    yoy = _compute_yoy(cpi)
    out["inflation_yoy"] = _align_series(yoy, out["date"])
    out["fed_funds"] = _align_series(fed_funds, out["date"])
    out["vix"] = _align_series(vix, out["date"])

    out["rates_change_6m"] = out["fed_funds"] - out["fed_funds"].shift(126)

    out["high_inflation"] = out["inflation_yoy"] >= 0.03
    out["rising_rates"] = out["rates_change_6m"] >= 0.005
    out["risk_off"] = out["vix"] >= 20.0

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _align_series(df: pd.DataFrame, dates: pd.Series) -> pd.Series:
    if df.empty:
        return pd.Series([None] * len(dates), index=dates.index)
    series = df.set_index("date")["value"].sort_index()
    series = series.reindex(pd.to_datetime(dates), method="ffill")
    return series.values


def _compute_yoy(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.DataFrame()
    series = df.set_index("date")["value"].sort_index()
    yoy = series.pct_change(12)
    out = yoy.reset_index()
    out.columns = ["date", "value"]
    return out
