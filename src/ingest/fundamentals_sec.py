from __future__ import annotations

import json
import pathlib
import sys
import time

import pandas as pd
import requests

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.utils_io import (
    PARQ,
    ROOT,
    SEC_CACHE,
    db_conn,
    get_ticker_cik_map,
    load_yaml,
    register_temp_view,
    sec_user_agent,
    today_str,
    unregister_temp_view,
)
from src.utils_io import normalize_ticker, yahoo_ticker, sec_ticker

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


def _load_universe_tickers() -> pd.DataFrame:
    uni_path = ROOT / "data" / "universe.csv"
    if uni_path.exists():
        df = pd.read_csv(uni_path)
        if "vendor_ticker" not in df.columns:
            df["vendor_ticker"] = df["ticker"].apply(yahoo_ticker)
        df["ticker"] = df["ticker"].apply(normalize_ticker)
        df["vendor_ticker"] = df["vendor_ticker"].apply(sec_ticker)
        return df[["ticker", "vendor_ticker"]]
    watch = load_yaml(ROOT / "watchlist.yml") or {}
    tickers = [normalize_ticker(t) for t in watch.get("tickers", [])]
    return pd.DataFrame({"ticker": tickers, "vendor_ticker": [sec_ticker(t) for t in tickers]})


def pull_company_facts(cik: str, cache_hours: int, retries: int, backoff: int) -> dict:
    cache_path = SEC_CACHE / f"companyfacts_{cik}.json"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            return json.loads(cache_path.read_text(encoding="utf-8"))

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": sec_user_agent()}
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            cache_path.write_bytes(resp.content)
            return resp.json()
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    resp.raise_for_status()
    return {}


def extract_quarterly(facts: dict, ticker: str) -> pd.DataFrame:
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    rows = {}
    for name, tag in FACTS.items():
        series = usgaap.get(tag, {}).get("units", {})
        vals = series.get("USD") or next(iter(series.values()), [])
        out = []
        for v in vals:
            if "frame" in v or "end" in v:
                out.append(v)
        rows[name] = out

    periods = {}
    for name, out in rows.items():
        for v in out:
            end = v.get("end") or v.get("fy")
            if not end:
                continue
            entry = periods.setdefault(end, {})
            entry[name] = v.get("val")
            if "filed" in v:
                entry.setdefault("filed", v.get("filed"))

    df = pd.DataFrame([
        {"fiscal_end": k, **vals} for k, vals in periods.items()
    ])
    if not df.empty:
        df["ticker"] = ticker
        df["fiscal_end"] = pd.to_datetime(df["fiscal_end"])
        if "filed" in df.columns:
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        entity = facts.get("entity", {})
        df["cik"] = entity.get("cik")
        df["sic"] = str(entity.get("sic") or "")
        df["entity_name"] = entity.get("name")
        df = df.sort_values("fiscal_end")
    return df


def main() -> None:
    cfg = load_yaml(ROOT / "config.yml") or {}
    fetch_cfg = cfg.get("fetch", {})
    retries = int(fetch_cfg.get("retries", 3))
    backoff = int(fetch_cfg.get("backoff_seconds", 2))
    cache_hours = int(fetch_cfg.get("sec_cache_hours", 168))

    watch = load_yaml(ROOT / "watchlist.yml") or {}
    overrides = {normalize_ticker(k): v for k, v in (watch.get("cik_overrides", {}) or {}).items()}

    universe = _load_universe_tickers()
    cik_map = get_ticker_cik_map()
    frames: list[pd.DataFrame] = []
    tickers = []
    for _, row in universe.iterrows():
        t = normalize_ticker(row["ticker"])
        vendor = sec_ticker(row["vendor_ticker"])
        tickers.append(t)
        cik = overrides.get(t) or cik_map.get(t) or cik_map.get(vendor)
        if not cik:
            print(f"[WARN] Missing CIK for {t} (vendor {vendor}); skipping.")
            continue
        facts = pull_company_facts(cik, cache_hours, retries, backoff)
        df = extract_quarterly(facts, t)
        if not df.empty:
            df["cik"] = df["cik"].fillna(cik)
            frames.append(df)
        time.sleep(0.2)

    fundamentals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["fiscal_end", "ticker", *FACTS.keys(), "filed", "cik", "sic", "entity_name"]
    )

    expected_cols = ["fiscal_end", "ticker", *FACTS.keys(), "filed", "cik", "sic", "entity_name"]
    fundamentals = fundamentals.reindex(columns=expected_cols)

    con = db_conn()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
            fiscal_end DATE, ticker VARCHAR,
            Revenue DOUBLE, NetIncome DOUBLE, SharesDiluted DOUBLE,
            OperatingCF DOUBLE, CapitalExpenditures DOUBLE, TotalAssets DOUBLE,
            TotalLiabilities DOUBLE, CashAndEquivalents DOUBLE, Debt DOUBLE,
            GrossProfit DOUBLE, CurrentAssets DOUBLE, CurrentLiabilities DOUBLE,
            EBITDA DOUBLE, InterestExpense DOUBLE,
            filed TIMESTAMP, cik VARCHAR, sic VARCHAR, entity_name VARCHAR
        )
        """
    )
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
