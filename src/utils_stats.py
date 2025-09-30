import numpy as np
import pandas as pd

def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean(skipna=True)) / series.std(skipna=True)

def winsorize(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lower=lo, upper=hi)

def pct_change_n(prices: pd.Series, n: int) -> float:
    if len(prices) < n+1 or prices.iloc[-n-1] == 0:
        return np.nan
    return prices.iloc[-1] / prices.iloc[-n-1] - 1.0
