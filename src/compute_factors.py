import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src import industry_map
from src.utils_io import (
    PARQ,
    ROOT,
    db_conn,
    load_yaml,
    register_temp_view,
    today_str,
    unregister_temp_view,
)
from src.utils_stats import (
    calc_ev_to_ebitda,
    calc_fcf_yield,
    calc_roic,
    industry_zscores,
    pct_change_n,
    piotroski_f_score,
    winsorize,
    zscore,
)


def _map_industry(sic: str) -> str:
    code = str(sic or "").strip()
    if not code:
        return industry_map.DEFAULT
    if code in industry_map.SIC_TO_INDUSTRY:
        return industry_map.SIC_TO_INDUSTRY[code]
    if len(code) >= 3 and code[:3] in industry_map.SIC_TO_INDUSTRY:
        return industry_map.SIC_TO_INDUSTRY[code[:3]]
    if len(code) >= 2 and code[:2] in industry_map.SIC_TO_INDUSTRY:
        return industry_map.SIC_TO_INDUSTRY[code[:2]]
    return industry_map.DEFAULT


def _build_ttm_rollup(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("fiscal_end").copy()
    res = pd.DataFrame({"fiscal_end": g["fiscal_end"].values})
    rolling_cols = {
        "RevenueTTM": "Revenue",
        "NetIncomeTTM": "NetIncome",
        "OpCFTTM": "OperatingCF",
        "CapexTTM": "CapitalExpenditures",
        "GrossProfitTTM": "GrossProfit",
        "EBITDATTM": "EBITDA",
        "InterestExpenseTTM": "InterestExpense",
    }
    for out_col, src_col in rolling_cols.items():
        res[out_col] = g[src_col].rolling(4, min_periods=1).sum().values

    res["SharesDilutedTTM"] = g["SharesDiluted"].rolling(4, min_periods=1).mean().values
    passthrough_cols = [
        "Debt",
        "CashAndEquivalents",
        "TotalAssets",
        "TotalLiabilities",
        "CurrentAssets",
        "CurrentLiabilities",
        "filed",
        "sic",
        "cik",
        "entity_name",
    ]
    for col in passthrough_cols:
        res[col] = g[col].values
    res["ticker"] = g["ticker"].values
    res["FCFTTM"] = res["OpCFTTM"] - res["CapexTTM"].fillna(0)
    return res


def compute(prices: pd.DataFrame, fnds: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or fnds.empty:
        return pd.DataFrame()

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    fnds = fnds.copy()
    fnds["fiscal_end"] = pd.to_datetime(fnds["fiscal_end"])
    fnds = fnds.sort_values(["ticker", "fiscal_end"])

    rollup = (
        fnds.groupby("ticker", group_keys=False)
        .apply(_build_ttm_rollup)
        .reset_index(drop=True)
    )

    piotroski_series = []
    for ticker, grp in rollup.groupby("ticker"):
        scores = piotroski_f_score(grp)
        piotroski_series.append(pd.Series(scores.values, index=grp.index))
    if piotroski_series:
        rollup["PiotroskiF"] = pd.concat(piotroski_series).sort_index()
    else:
        rollup["PiotroskiF"] = np.nan

    latest = rollup.groupby("ticker").tail(1).copy()

    last_price = (
        prices.groupby("ticker").tail(1)[["ticker", "adj_close"]].rename(columns={"adj_close": "Price"})
    )
    latest = latest.merge(last_price, on="ticker", how="left")

    latest["EPS_TTM"] = latest["NetIncomeTTM"] / latest["SharesDilutedTTM"].replace({0: np.nan})
    latest["PE_TTM"] = latest["Price"] / latest["EPS_TTM"].replace({0: np.nan})
    latest["FCFYield_TTM"] = calc_fcf_yield(
        latest["FCFTTM"], latest["Price"], latest["SharesDilutedTTM"]
    )
    latest["EVToEBITDA"] = calc_ev_to_ebitda(
        latest["Price"],
        latest["SharesDilutedTTM"],
        latest["Debt"],
        latest["CashAndEquivalents"],
        latest["EBITDATTM"],
    )
    latest["ROIC"] = calc_roic(
        latest["NetIncomeTTM"],
        latest["InterestExpenseTTM"],
        latest["TotalAssets"],
        latest["CurrentLiabilities"],
        latest["CashAndEquivalents"],
        latest["Debt"],
    )
    latest["Leverage"] = (
        (latest["Debt"].fillna(0) - latest["CashAndEquivalents"].fillna(0))
        / latest["TotalAssets"].replace({0: np.nan})
    )
    latest["Qual_ROA"] = latest["NetIncomeTTM"] / latest["TotalAssets"].replace({0: np.nan})

    mom6 = prices.groupby("ticker").apply(lambda g: pct_change_n(g["adj_close"], 126))
    mom12 = prices.groupby("ticker").apply(lambda g: pct_change_n(g["adj_close"], 252))
    momentum = pd.DataFrame({"ticker": mom6.index, "Mom6m": mom6.values, "Mom12m": mom12.values})
    latest = latest.merge(momentum, on="ticker", how="left")

    returns = prices.copy()
    returns["ret"] = returns.groupby("ticker")["adj_close"].pct_change()

    def vol_30(series: pd.Series) -> float:
        tail = series.dropna().tail(30)
        if len(tail) < 20:
            return np.nan
        return tail.std(ddof=0) * np.sqrt(252)

    def sharpe_1y(series: pd.Series) -> float:
        tail = series.dropna().tail(252)
        if len(tail) < 200:
            return np.nan
        std = tail.std(ddof=0)
        if std == 0 or np.isnan(std):
            return np.nan
        return np.sqrt(252) * tail.mean() / std

    vol = returns.groupby("ticker")["ret"].apply(vol_30)
    sharpe = returns.groupby("ticker")["ret"].apply(sharpe_1y)
    risk = pd.DataFrame({
        "ticker": vol.index,
        "Volatility30d": vol.values,
        "Sharpe1y": sharpe.values,
    })
    latest = latest.merge(risk, on="ticker", how="left")

    pe_positive = latest["PE_TTM"].where(latest["PE_TTM"] > 0)
    latest["Value_PE"] = -zscore(winsorize(np.log(pe_positive)))
    latest["Value_FCF"] = zscore(winsorize(latest["FCFYield_TTM"]))
    evebitda_positive = latest["EVToEBITDA"].where(latest["EVToEBITDA"] > 0)
    latest["Value_EVEBITDA"] = -zscore(winsorize(np.log(evebitda_positive)))
    latest["ValueScore_raw"] = latest[["Value_PE", "Value_FCF", "Value_EVEBITDA"]].mean(axis=1, skipna=True)

    latest["Quality_ROIC"] = zscore(winsorize(latest["ROIC"]))
    latest["Quality_Pio"] = zscore(winsorize(latest["PiotroskiF"]))
    latest["QualityScore_raw"] = latest[["Quality_ROIC", "Quality_Pio"]].mean(axis=1, skipna=True)

    latest["Mom6m_z"] = zscore(winsorize(latest["Mom6m"]))
    latest["Mom12m_z"] = zscore(winsorize(latest["Mom12m"]))
    latest["MomScore"] = latest[["Mom6m_z", "Mom12m_z"]].mean(axis=1, skipna=True)

    latest["industry"] = latest["sic"].apply(_map_industry)
    latest["ValueScore"] = industry_zscores(latest["ValueScore_raw"], latest["industry"])
    latest["QualityScore"] = industry_zscores(latest["QualityScore_raw"], latest["industry"])

    cfg = load_yaml(ROOT / "config.yml") or {}
    weights_cfg = cfg.get("weights", {})
    default_weights = {"value": 0.4, "quality": 0.4, "momentum": 0.2}
    weights = {
        "value": float(weights_cfg.get("value", default_weights["value"])),
        "quality": float(weights_cfg.get("quality", default_weights["quality"])),
        "momentum": float(weights_cfg.get("momentum", default_weights["momentum"])),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = default_weights
        total_weight = 1.0
    weights = {k: v / total_weight for k, v in weights.items()}

    latest["Composite"] = (
        weights["value"] * latest["ValueScore"]
        + weights["quality"] * latest["QualityScore"]
        + weights["momentum"] * latest["MomScore"]
    )

    latest["CompositePctile"] = latest["Composite"].rank(pct=True)
    latest["ValueZ"] = latest["ValueScore"]
    latest["QualityZ"] = latest["QualityScore"]

    cols = [
        "ticker",
        "Price",
        "PE_TTM",
        "FCFYield_TTM",
        "EVToEBITDA",
        "ROIC",
        "Leverage",
        "Qual_ROA",
        "PiotroskiF",
        "Mom6m",
        "Mom12m",
        "Volatility30d",
        "Sharpe1y",
        "ValueScore",
        "QualityScore",
        "MomScore",
        "Composite",
        "CompositePctile",
        "ValueZ",
        "QualityZ",
        "industry",
        "sic",
        "filed",
        "cik",
        "entity_name",
    ]

    return latest[cols].sort_values("Composite", ascending=False)


def main():
    con = db_conn()
    prices = con.execute("SELECT * FROM prices_daily").df()
    fnds = con.execute("SELECT * FROM fundamentals_quarterly").df()
    scores = compute(prices, fnds)
    scores.to_parquet(PARQ / f"scores_daily_{today_str()}.parquet", index=False)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scores_daily (
            ticker VARCHAR,
            Price DOUBLE,
            PE_TTM DOUBLE,
            FCFYield_TTM DOUBLE,
            EVToEBITDA DOUBLE,
            ROIC DOUBLE,
            Leverage DOUBLE,
            Qual_ROA DOUBLE,
            PiotroskiF DOUBLE,
            Mom6m DOUBLE,
            Mom12m DOUBLE,
            Volatility30d DOUBLE,
            Sharpe1y DOUBLE,
            ValueScore DOUBLE,
            QualityScore DOUBLE,
            MomScore DOUBLE,
            Composite DOUBLE,
            CompositePctile DOUBLE,
            ValueZ DOUBLE,
            QualityZ DOUBLE,
            industry VARCHAR,
            sic VARCHAR,
            filed TIMESTAMP,
            cik VARCHAR,
            entity_name VARCHAR
        )
        """
    )
    con.execute("DELETE FROM scores_daily")
    view_name = register_temp_view(con, "scores_tmp", scores)
    if view_name:
        con.execute(f"INSERT INTO scores_daily SELECT * FROM {view_name}")
    unregister_temp_view(con, view_name)
    print("Computed scores:", len(scores))


if __name__ == "__main__":
    main()
