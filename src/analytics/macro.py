from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class MacroPayload:
    status: str
    missing_series: list[str]
    as_of: str | None
    flags: pd.DataFrame


def compute_macro_regime_payload(
    dates: pd.DatetimeIndex,
    cpi: pd.DataFrame,
    fed_funds: pd.DataFrame,
    vix: pd.DataFrame,
) -> MacroPayload:
    if dates.empty:
        return MacroPayload(status="unavailable", missing_series=["CPIAUCSL", "DFF", "VIXCLS"], as_of=None, flags=pd.DataFrame())

    base = pd.DataFrame({"date": pd.to_datetime(dates)}).dropna().drop_duplicates().sort_values("date")

    out = base.copy()
    missing = []
    yoy = _compute_yoy(cpi)
    if yoy.empty:
        missing.append("CPIAUCSL")
        out["inflation_yoy"] = None
    else:
        out["inflation_yoy"] = _align_series(yoy, out["date"])

    if fed_funds.empty:
        missing.append("DFF")
        out["fed_funds"] = None
    else:
        out["fed_funds"] = _align_series(fed_funds, out["date"])

    if vix.empty:
        missing.append("VIXCLS")
        out["vix"] = None
    else:
        out["vix"] = _align_series(vix, out["date"])

    if "fed_funds" in out.columns and out["fed_funds"].notna().sum() >= 127:
        out["rates_change_6m"] = out["fed_funds"] - out["fed_funds"].shift(126)
    else:
        out["rates_change_6m"] = None
        if "DFF" not in missing:
            missing.append("DFF")

    out["high_inflation"] = out["inflation_yoy"] >= 0.03 if out["inflation_yoy"].notna().any() else None
    out["rising_rates"] = out["rates_change_6m"] >= 0.005 if out["rates_change_6m"].notna().any() else None
    out["risk_off"] = out["vix"] >= 20.0 if out["vix"].notna().any() else None

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    flags = out
    as_of = flags["date"].iloc[-1] if not flags.empty else None

    if missing and flags[["high_inflation", "rising_rates", "risk_off"]].isna().all(axis=None):
        return MacroPayload(status="unavailable", missing_series=sorted(set(missing)), as_of=as_of, flags=pd.DataFrame())
    if missing:
        return MacroPayload(status="partial", missing_series=sorted(set(missing)), as_of=as_of, flags=flags)
    return MacroPayload(status="ok", missing_series=[], as_of=as_of, flags=flags)


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
    if series.dropna().shape[0] < 13:
        return pd.DataFrame()
    yoy = series.pct_change(12)
    out = yoy.reset_index()
    out.columns = ["date", "value"]
    return out
