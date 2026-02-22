from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_metrics(
    performance: pd.DataFrame | pd.Series,
    window: int = 63,
    risk_free_series: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    if isinstance(performance, pd.Series):
        performance = pd.DataFrame({"date": performance.index, "daily_return": performance.values})

    if performance.empty or "daily_return" not in performance.columns:
        return pd.DataFrame()

    perf = performance.copy()
    perf["date"] = pd.to_datetime(perf["date"], errors="coerce")
    perf = perf.dropna(subset=["date"])
    perf = perf.sort_values("date")

    returns = pd.to_numeric(perf["daily_return"], errors="coerce").fillna(0.0)
    rf_daily: pd.Series | None = None
    if isinstance(risk_free_series, pd.Series):
        rf = pd.DataFrame({"date": risk_free_series.index, "rf_daily_return": risk_free_series.values})
        rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
        rf = rf.dropna(subset=["date"])
        aligned = perf.set_index("date")[["daily_return"]].join(
            rf.set_index("date")[["rf_daily_return"]],
            how="left",
        )
        daily = pd.to_numeric(aligned["daily_return"], errors="coerce")
        rf_daily = pd.to_numeric(aligned["rf_daily_return"], errors="coerce")
        returns = (daily - rf_daily).where(rf_daily.notna()).reset_index(drop=True)
    elif risk_free_series is not None and not risk_free_series.empty and "date" in risk_free_series.columns:
        rf = risk_free_series.copy()
        rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
        rf = rf.dropna(subset=["date", "rf_daily_return"])
        aligned = perf.set_index("date")[["daily_return"]].join(
            rf.set_index("date")[["rf_daily_return"]],
            how="left",
        )
        daily = pd.to_numeric(aligned["daily_return"], errors="coerce")
        rf_daily = pd.to_numeric(aligned["rf_daily_return"], errors="coerce")
        returns = (daily - rf_daily).where(rf_daily.notna()).reset_index(drop=True)

    rf_coverage_pct = float(rf_daily.notna().mean()) if rf_daily is not None else None

    rolling_vol = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_sharpe = (rolling_mean * 252.0) / rolling_vol.replace({0.0: np.nan})
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    if "drawdown" in perf.columns:
        drawdown = pd.to_numeric(perf["drawdown"], errors="coerce")
    else:
        drawdown = pd.Series(np.nan, index=perf.index, dtype=float)
    rolling_drawdown = drawdown.rolling(window, min_periods=window).min()

    out = pd.DataFrame(
        {
            "date": perf["date"].dt.strftime("%Y-%m-%d"),
            "rolling_volatility": rolling_vol.values,
            "rolling_sharpe": rolling_sharpe.values,
            "rolling_drawdown": rolling_drawdown.values,
            "rf_coverage_pct": rf_coverage_pct,
        }
    )
    return out
