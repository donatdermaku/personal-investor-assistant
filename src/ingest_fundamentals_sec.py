import json, time, pathlib, requests, pandas as pd
from src.utils_io import ROOT, PARQ, db_conn, load_yaml, today_str

# Minimal static mapping; extend as needed
TICKER_TO_CIK = {
    "MSFT":"0000789019",
    "AAPL":"0000320193",
    "AMZN":"0001018724",
    "GOOGL":"0001652044",
}

FACTS = {
    "Revenue":"Revenues",                    # may map to "RevenueFromContractWithCustomerExcludingAssessedTax"
    "NetIncome":"NetIncomeLoss",
    "SharesDiluted":"WeightedAverageNumberOfDilutedSharesOutstanding",
    "OperatingCF":"NetCashProvidedByUsedInOperatingActivities",
    "CapitalExpenditures":"PaymentsToAcquirePropertyPlantAndEquipment",
    "TotalAssets":"Assets",
    "TotalLiabilities":"Liabilities",
    "CashAndEquivalents":"CashAndCashEquivalentsAtCarryingValue",
    "Debt":"LongTermDebtNoncurrent",
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
            periods.setdefault(end, {})[name]=v.get("val")

    df = pd.DataFrame([
        {"fiscal_end": k, **vals} for k, vals in periods.items()
    ])
    if not df.empty:
        df["ticker"]=ticker
        df["fiscal_end"]=pd.to_datetime(df["fiscal_end"])
        df = df.sort_values("fiscal_end")
    return df

def main():
    cfg = load_yaml(ROOT / "watchlist.yml")
    tickers = cfg["tickers"]
    frames=[]
    for t in tickers:
        cik = TICKER_TO_CIK.get(t)
        if not cik:
            continue
        facts = pull_company_facts(cik)
        df = extract_quarterly(facts, t)
        if not df.empty:
            frames.append(df)

    fundamentals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["fiscal_end","ticker", *FACTS.keys()]
    )

    con = db_conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
            fiscal_end DATE, ticker VARCHAR,
            Revenue DOUBLE, NetIncome DOUBLE, SharesDiluted DOUBLE,
            OperatingCF DOUBLE, CapitalExpenditures DOUBLE, TotalAssets DOUBLE,
            TotalLiabilities DOUBLE, CashAndEquivalents DOUBLE, Debt DOUBLE
        )
    """)
    # upsert by ticker+fiscal_end (simple replace approach)
    con.execute("DELETE FROM fundamentals_quarterly WHERE ticker IN ({})".format(
        ",".join(["?"]*len(tickers))), tickers)
    con.execute("INSERT INTO fundamentals_quarterly SELECT * FROM df", {"df": fundamentals})

    from utils_io import write_parquet
    write_parquet(fundamentals, PARQ / f"fundamentals_quarterly_{today_str()}.parquet")
    print(f"Saved fundamentals for {len(tickers)} tickers.")

if __name__ == "__main__":
    main()
