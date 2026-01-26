from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market_data.fred import load_cached_series


def load_cached_fred_series(series_id: str) -> pd.DataFrame:
    df = load_cached_series(series_id)
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    return df.sort_values("date")


@dataclass
class MacroPayload:
    status: str
    available_series: list[str]
    missing_series: list[str]
    tags: list[str]
    warnings: list[str]
    as_of: str | None
    cache_status: dict[str, str]
    flags: pd.DataFrame


def compute_macro_regime_payload(
    dates: pd.DatetimeIndex,
    cpi: pd.DataFrame,
    fed_funds: pd.DataFrame,
    vix: pd.DataFrame,
    *,
    cache_status: dict[str, str] | None = None,
) -> MacroPayload:
    if dates.empty:
        return MacroPayload(
            status="unavailable",
            available_series=[],
            missing_series=["CPIAUCSL", "DFF", "VIXCLS"],
            tags=[],
            warnings=["No dates available for macro context."],
            as_of=None,
            cache_status=cache_status or {},
            flags=pd.DataFrame(),
        )

    base = pd.DataFrame({"date": pd.to_datetime(dates)}).dropna().drop_duplicates().sort_values("date")

    out = base.copy()
    missing: list[str] = []
    available: list[str] = []
    yoy = _compute_yoy(cpi)
    if yoy.empty:
        missing.append("CPIAUCSL")
        out["inflation_yoy"] = None
    else:
        out["inflation_yoy"] = _align_series(yoy, out["date"])
        available.append("CPIAUCSL")

    if fed_funds.empty:
        missing.append("DFF")
        out["fed_funds"] = None
    else:
        out["fed_funds"] = _align_series(fed_funds, out["date"])
        available.append("DFF")

    if vix.empty:
        missing.append("VIXCLS")
        out["vix"] = None
    else:
        out["vix"] = _align_series(vix, out["date"])
        available.append("VIXCLS")

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

    tags = []
    warnings: list[str] = []
    if not flags.empty:
        latest = flags.iloc[-1]
        if latest.get("high_inflation") is True:
            tags.append("high inflation")
        if latest.get("rising_rates") is True:
            tags.append("rising rates")
        if latest.get("risk_off") is True:
            tags.append("risk-off")

    if "VIXCLS" in missing:
        warnings.append("VIX unavailable; risk-off tag suppressed.")
    if "CPIAUCSL" in missing:
        warnings.append("CPI unavailable; inflation tag suppressed.")
    if "DFF" in missing:
        warnings.append("Fed Funds unavailable; rate tags suppressed.")

    if missing and flags[["high_inflation", "rising_rates", "risk_off"]].isna().all(axis=None):
        return MacroPayload(
            status="unavailable",
            available_series=sorted(set(available)),
            missing_series=sorted(set(missing)),
            tags=[],
            warnings=warnings,
            as_of=as_of,
            cache_status=cache_status or {},
            flags=pd.DataFrame(),
        )
    if missing:
        return MacroPayload(
            status="partial",
            available_series=sorted(set(available)),
            missing_series=sorted(set(missing)),
            tags=tags,
            warnings=warnings,
            as_of=as_of,
            cache_status=cache_status or {},
            flags=flags,
        )
    return MacroPayload(
        status="sufficient",
        available_series=sorted(set(available)),
        missing_series=[],
        tags=tags,
        warnings=[],
        as_of=as_of,
        cache_status=cache_status or {},
        flags=flags,
    )


def _align_series(df: pd.DataFrame, dates: pd.Series) -> pd.Series:
    if df.empty:
        return pd.Series([None] * len(dates), index=dates.index)
    series = df.set_index("date")["value"].sort_index()
    # Ensure series index is timezone-naive DatetimeIndex
    if hasattr(series.index, 'tz') and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = pd.to_datetime(series.index, errors="coerce")
    # Convert target dates to timezone-naive DatetimeIndex
    target_dates = pd.to_datetime(dates, errors="coerce")
    if hasattr(target_dates, 'tz') and target_dates.tz is not None:
        target_dates = target_dates.tz_localize(None)
    series = series.reindex(target_dates, method="ffill")
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
