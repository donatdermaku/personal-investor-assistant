import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(skipna=True, ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean(skipna=True)) / std


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if series.empty:
        return series
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def pct_change_n(prices: pd.Series, n: int) -> float:
    if len(prices) < n + 1 or prices.iloc[-n - 1] == 0:
        return np.nan
    return prices.iloc[-1] / prices.iloc[-n - 1] - 1.0


def industry_zscores(values: pd.Series, industries: pd.Series) -> pd.Series:
    def _zs(x: pd.Series) -> pd.Series:
        return zscore(x.fillna(x.mean(skipna=True)))

    grouped = values.groupby(industries)
    return grouped.transform(_zs)


def calc_fcf_yield(fcf_ttm: pd.Series, price: pd.Series, shares: pd.Series) -> pd.Series:
    denominator = price * shares
    denominator = denominator.replace({0: np.nan})
    return fcf_ttm / denominator


def calc_ev_to_ebitda(
    price: pd.Series,
    shares: pd.Series,
    debt: pd.Series,
    cash: pd.Series,
    ebitda_ttm: pd.Series,
) -> pd.Series:
    enterprise_value = price * shares + debt.fillna(0) - cash.fillna(0)
    return enterprise_value / ebitda_ttm.replace({0: np.nan})


def calc_roic(
    net_income_ttm: pd.Series,
    interest_expense_ttm: pd.Series,
    total_assets: pd.Series,
    current_liabilities: pd.Series,
    cash: pd.Series,
    debt: pd.Series,
) -> pd.Series:
    invested_capital = (
        total_assets.fillna(0)
        - current_liabilities.fillna(0)
        - cash.fillna(0)
        + debt.fillna(0)
    )
    invested_capital = invested_capital.replace({0: np.nan})
    nopat = net_income_ttm.fillna(0) + interest_expense_ttm.fillna(0)
    return nopat / invested_capital


def piotroski_f_score(df: pd.DataFrame) -> pd.Series:
    required = [
        "fiscal_end",
        "NetIncomeTTM",
        "OpCFTTM",
        "TotalAssets",
        "Debt",
        "CurrentAssets",
        "CurrentLiabilities",
        "GrossProfitTTM",
        "RevenueTTM",
        "SharesDilutedTTM",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' for Piotroski F-score computation")

    df = df.sort_values("fiscal_end")
    roa = df["NetIncomeTTM"] / df["TotalAssets"].replace({0: np.nan})
    cfo = df["OpCFTTM"] / df["TotalAssets"].replace({0: np.nan})
    accrual = df["OpCFTTM"] - df["NetIncomeTTM"]
    leverage = df["Debt"] / df["TotalAssets"].replace({0: np.nan})
    current_ratio = df["CurrentAssets"] / df["CurrentLiabilities"].replace({0: np.nan})
    gross_margin = df["GrossProfitTTM"] / df["RevenueTTM"].replace({0: np.nan})
    asset_turnover = df["RevenueTTM"] / df["TotalAssets"].replace({0: np.nan})
    shares = df["SharesDilutedTTM"]

    components = pd.DataFrame(index=df.index)
    components["roa_pos"] = (roa > 0).astype(int)
    components["cfo_pos"] = (df["OpCFTTM"] > 0).astype(int)
    components["delta_roa"] = (roa > roa.shift(1)).astype(int)
    components["accrual"] = (accrual > 0).astype(int)
    components["delta_leverage"] = (leverage < leverage.shift(1)).astype(int)
    components["delta_liquidity"] = (current_ratio > current_ratio.shift(1)).astype(int)
    components["margin"] = (gross_margin > gross_margin.shift(1)).astype(int)
    components["turnover"] = (asset_turnover > asset_turnover.shift(1)).astype(int)
    components["shares"] = (shares <= shares.shift(1)).astype(int)

    return components.fillna(0).sum(axis=1)
