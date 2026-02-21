from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_metrics(
    performance: pd.DataFrame,
    window: int = 63,
    risk_free_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if performance.empty or "daily_return" not in performance.columns:
        return pd.DataFrame()

    perf = performance.copy()
    perf["date"] = pd.to_datetime(perf["date"], errors="coerce")
    perf = perf.dropna(subset=["date"])
    perf = perf.sort_values("date")

    returns = pd.to_numeric(perf["daily_return"], errors="coerce").fillna(0.0)
    if risk_free_series is not None and not risk_free_series.empty and "date" in risk_free_series.columns:
        rf = risk_free_series.copy()
        rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
        rf = rf.dropna(subset=["date", "rf_daily_return"])
        aligned = perf.set_index("date")[["daily_return"]].join(
            rf.set_index("date")[["rf_daily_return"]],
            how="left",
        )
        returns = (aligned["daily_return"] - aligned["rf_daily_return"].fillna(0.0)).reset_index(drop=True)
    rolling_vol = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_sharpe = (rolling_mean * 252.0) / rolling_vol.replace({0.0: np.nan})
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    drawdown = pd.to_numeric(perf.get("drawdown"), errors="coerce")
    rolling_drawdown = drawdown.rolling(window, min_periods=window).min()

    out = pd.DataFrame(
        {
            "date": perf["date"].dt.strftime("%Y-%m-%d"),
            "rolling_volatility": rolling_vol.values,
            "rolling_sharpe": rolling_sharpe.values,
            "rolling_drawdown": rolling_drawdown.values,
        }
    )
    return out
