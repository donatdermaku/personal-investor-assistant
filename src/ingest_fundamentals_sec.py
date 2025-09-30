import pathlib
import sys
import time

import pandas as pd
import requests

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils_io import (
    PARQ,
    ROOT,
    db_conn,
    get_ticker_cik_map,
    load_yaml,
    register_temp_view,
    today_str,
    unregister_temp_view,
)

FACTS = {
    "Revenue": "Revenues",
    "NetIncome": "NetIncomeLoss",
    "SharesDiluted": "WeightedAverageNumberOfDilutedSharesOutstanding",
    "OperatingCF": "NetCashProvidedByUsedInOperatingActivities",
    "CapitalExpenditures": "PaymentsToAcquirePropertyPlantAndEquipment",
    "TotalAssets": "Assets",
    "TotalLiabilities": "Liabilities",
    "CashAndEquivalents": "CashAndCashEquivalentsAtCarryingValue",
    "Debt": "LongTermDebtNoncurrent",
    "GrossProfit": "GrossProfit",
    "CurrentAssets": "AssetsCurrent",
    "CurrentLiabilities": "LiabilitiesCurrent",
    "EBITDA": "EarningsBeforeInterestTaxesDepreciationAmortization",
    "InterestExpense": "InterestExpense",
}

UA = {"User-Agent": "personal-investor-assistant (contact: example@example.com)"}

def pull_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for _ in range(3):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200: return r.json()
        time.sleep(2)
    r.raise_for_status()

def extract_quarterly(facts: dict, ticker: str) -> pd.DataFrame:
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    rows = {}
    for name, tag in FACTS.items():
        series = usgaap.get(tag, {}).get("units", {})
        # prefer USD units
        vals = series.get("USD") or next(iter(series.values()), [])
        # pick only quarterly/instant values
        out=[]
        for v in vals:
            # filter by frame (e.g., "CY2024Q4") or period; keep last 4Y
            if "frame" in v or "end" in v:
                out.append(v)
        rows[name]=out

    # normalize into period rows by 'end' date
    periods = {}
    for name, out in rows.items():
        for v in out:
            end = v.get("end") or v.get("fy")  # end is ISO date
            if not end: continue
            entry = periods.setdefault(end, {})
            entry[name]=v.get("val")
            if "filed" in v:
                entry.setdefault("filed", v.get("filed"))

    df = pd.DataFrame([
        {"fiscal_end": k, **vals} for k, vals in periods.items()
    ])
    if not df.empty:
        df["ticker"]=ticker
        df["fiscal_end"]=pd.to_datetime(df["fiscal_end"])
        if "filed" in df.columns:
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        entity = facts.get("entity", {})
        df["cik"] = entity.get("cik")
        df["sic"] = str(entity.get("sic") or "")
        df["entity_name"] = entity.get("name")
        df = df.sort_values("fiscal_end")
    return df

def main():
    cfg = load_yaml(ROOT / "watchlist.yml")
    tickers = cfg["tickers"]
    overrides = {k.upper(): v for k, v in cfg.get("cik_overrides", {}).items()}
    cik_map = get_ticker_cik_map()
    frames=[]
    for t in tickers:
        cik = overrides.get(t.upper()) or cik_map.get(t.upper())
        if not cik:
            print(f"[WARN] Missing CIK for {t}; skipping.")
            continue
        facts = pull_company_facts(cik)
        df = extract_quarterly(facts, t)
        if not df.empty:
            df["cik"] = df["cik"].fillna(cik)
            frames.append(df)

    fundamentals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["fiscal_end","ticker", *FACTS.keys(), "filed", "cik", "sic", "entity_name"]
    )

    # Ensure the DataFrame has all expected columns (fill any missing with NaN)
    expected_cols = ["fiscal_end", "ticker", *FACTS.keys(), "filed", "cik", "sic", "entity_name"]
    fundamentals = fundamentals.reindex(columns=expected_cols)

    con = db_conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
            fiscal_end DATE, ticker VARCHAR,
            Revenue DOUBLE, NetIncome DOUBLE, SharesDiluted DOUBLE,
            OperatingCF DOUBLE, CapitalExpenditures DOUBLE, TotalAssets DOUBLE,
            TotalLiabilities DOUBLE, CashAndEquivalents DOUBLE, Debt DOUBLE,
            GrossProfit DOUBLE, CurrentAssets DOUBLE, CurrentLiabilities DOUBLE,
            EBITDA DOUBLE, InterestExpense DOUBLE,
            filed TIMESTAMP, cik VARCHAR, sic VARCHAR, entity_name VARCHAR
        )
    """)
    # upsert by ticker+fiscal_end (simple replace approach)
    if tickers:
        con.execute(
            "DELETE FROM fundamentals_quarterly WHERE ticker IN ({})".format(
                ",".join(["?"] * len(tickers))
            ),
            tickers,
        )
    view_name = register_temp_view(con, "fundamentals_tmp", fundamentals)
    if view_name:
        con.execute(f"INSERT INTO fundamentals_quarterly SELECT * FROM {view_name}")
    unregister_temp_view(con, view_name)

    from src.utils_io import write_parquet
    write_parquet(fundamentals, PARQ / f"fundamentals_quarterly_{today_str()}.parquet")
    print(f"Saved fundamentals for {len(tickers)} tickers.")

if __name__ == "__main__":
    main()
