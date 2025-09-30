import duckdb, pandas as pd, numpy as np
from src.utils_io import db_conn, PARQ, today_str
from src.utils_stats import zscore, winsorize

def compute(prices: pd.DataFrame, fnds: pd.DataFrame) -> pd.DataFrame:
    # TTM basics from fundamentals
    fnds = fnds.sort_values(["ticker","fiscal_end"])
    # forward-fill quarterly into TTM by group
    def ttm(series):
        return series.rolling(4, min_periods=1).sum()

    f = fnds.groupby("ticker", group_keys=False).apply(lambda g: pd.DataFrame({
        "fiscal_end": g["fiscal_end"],
        "RevenueTTM": ttm(g["Revenue"]),
        "NetIncomeTTM": ttm(g["NetIncome"]),
        "OpCFTTM": ttm(g["OperatingCF"]),
        "CapexTTM": ttm(g["CapitalExpenditures"]),
        "SharesDilutedTTM": g["SharesDiluted"].rolling(4, min_periods=1).mean(),
        "Debt": g["Debt"],
        "Cash": g["CashAndEquivalents"],
        "Assets": g["TotalAssets"],
        "Liabilities": g["TotalLiabilities"],
        "ticker": g["ticker"].values
    }))
    f["FCFTTM"] = f["OpCFTTM"] - (f["CapexTTM"].fillna(0))

    # last close per ticker
    p_last = prices.sort_values("date").groupby("ticker").tail(1)[["ticker","adj_close"]].rename(columns={"adj_close":"Price"})
    out = f.groupby("ticker").tail(1).merge(p_last, on="ticker", how="left")

    # simple ratios (guard against /0)
    out["EPS_TTM"] = out["NetIncomeTTM"] / out["SharesDilutedTTM"]
    out["PE_TTM"] = out["Price"] / out["EPS_TTM"].replace({0:np.nan})
    out["FCFYield_TTM"] = out["FCFTTM"] / (out["Price"] * out["SharesDilutedTTM"])

    # z-scores for Value (lower PE better; higher FCF yield better)
    out["val_pe"] = -zscore(winsorize(np.log(out["PE_TTM"])))
    out["val_fcf"] = zscore(winsorize(out["FCFYield_TTM"]))
    out["ValueScore"] = out[["val_pe","val_fcf"]].mean(axis=1, skipna=True)

    # Quality (gross/ROIC proxies minimal — keep simple with FCF/Assets and low leverage)
    out["Leverage"] = (out["Debt"] - out["Cash"]) / out["Assets"]
    out["Qual_ROA"] = out["NetIncomeTTM"] / out["Assets"]
    out["QualScore"] = zscore(winsorize(out["Qual_ROA"])) + (-zscore(winsorize(out["Leverage"])))
    out["QualScore"] = out["QualScore"] / 2.0

    # Momentum from prices (6m, 12m)
    pr = prices.sort_values("date")
    six = pr.groupby("ticker").apply(lambda g: g.tail(1)["adj_close"].iloc[0] / g.tail(126)["adj_close"].iloc[0] - 1 if len(g)>=127 else np.nan)
    twelve = pr.groupby("ticker").apply(lambda g: g.tail(1)["adj_close"].iloc[0] / g.tail(252)["adj_close"].iloc[0] - 1 if len(g)>=253 else np.nan)
    mom = pd.DataFrame({"ticker":six.index, "Mom6m":six.values, "Mom12m":twelve.values})
    out = out.merge(mom, on="ticker", how="left")
    out["MomScore"] = zscore(winsorize(out["Mom6m"].fillna(0.0))) * 0.5 + zscore(winsorize(out["Mom12m"].fillna(0.0))) * 0.5

    # Composite
    out["Composite"] = 0.4*out["ValueScore"] + 0.4*out["QualScore"] + 0.2*out["MomScore"]

    # select columns
    cols = ["ticker","Price","PE_TTM","FCFYield_TTM","Qual_ROA","Leverage","Mom6m","Mom12m",
            "ValueScore","QualScore","MomScore","Composite"]
    return out[cols].sort_values("Composite", ascending=False)

def main():
    con = db_conn()
    prices = con.execute("SELECT * FROM prices_daily").df()
    fnds = con.execute("SELECT * FROM fundamentals_quarterly").df()
    scores = compute(prices, fnds)
    scores.to_parquet(PARQ / f"scores_daily_{today_str()}.parquet", index=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scores_daily AS SELECT * FROM scores
    """)
    con.execute("DELETE FROM scores_daily")
    con.execute("INSERT INTO scores_daily SELECT * FROM scores", {"scores": scores})
    print("Computed scores:", len(scores))

if __name__ == "__main__":
    main()
