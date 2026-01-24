from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.macro import load_cached_fred_series


@dataclass
class RiskFreeSeries:
    series: pd.DataFrame
    status: str
    reason_codes: list[str]


def compute_risk_free_series(dates: pd.DatetimeIndex) -> RiskFreeSeries:
    if dates.empty:
        return RiskFreeSeries(series=pd.DataFrame(), status="unavailable", reason_codes=["NO_DATES"])
    fred = load_cached_fred_series("DTB3")
    if fred.empty:
        return RiskFreeSeries(series=pd.DataFrame(), status="unavailable", reason_codes=["MISSING_DTB3"])
    fred = fred.copy()
    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    fred = fred.dropna(subset=["date", "value"]).sort_values("date")
    if fred.empty:
        return RiskFreeSeries(series=pd.DataFrame(), status="unavailable", reason_codes=["INVALID_DTB3"])

    daily_dates = pd.to_datetime(dates).normalize().dropna().unique()
    rates = fred.set_index("date")["value"].reindex(daily_dates, method="ffill")
    if rates.isna().all():
        return RiskFreeSeries(series=pd.DataFrame(), status="unavailable", reason_codes=["NO_COVERAGE"])

    annual_rate = rates.fillna(method="ffill").fillna(0.0) / 100.0
    rf_daily = (1.0 + annual_rate) ** (1.0 / 252.0) - 1.0
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(daily_dates).strftime("%Y-%m-%d"),
            "rate": annual_rate.values,
            "rf_daily_return": rf_daily.values,
        }
    )
    return RiskFreeSeries(series=out, status="ok", reason_codes=[])
