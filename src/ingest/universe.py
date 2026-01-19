import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.utils_io import ROOT, UNIVERSE, fetch_url_cached, load_yaml, today_str, db_conn


SP500_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _load_watchlist() -> list[str]:
    watch = load_yaml(ROOT / "watchlist.yml") or {}
    tickers = watch.get("tickers", []) or []
    return [str(t).upper() for t in tickers]


def _fetch_sp500() -> pd.DataFrame:
    path = fetch_url_cached(SP500_URL, "sp500_constituents.csv", max_age_hours=24)
    df = pd.read_csv(path)
    df = df.rename(columns={"Symbol": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df[["ticker"]]


def _fetch_nasdaq100() -> pd.DataFrame:
    path = fetch_url_cached(NASDAQ100_URL, "nasdaq100.html", max_age_hours=24)
    tables = pd.read_html(path)
    if not tables:
        return pd.DataFrame(columns=["ticker"])
    # Wikipedia page includes a table with "Ticker" column.
    df = tables[0]
    df = df.rename(columns={"Ticker": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df[["ticker"]]


def resolve_universe(cfg: dict) -> pd.DataFrame:
    uni_cfg = cfg.get("universe", {}) if cfg else {}
    mode = str(uni_cfg.get("mode", "manual")).lower()
    if mode == "sp500":
        df = _fetch_sp500()
    elif mode == "nasdaq100":
        df = _fetch_nasdaq100()
    else:
        tickers = uni_cfg.get("tickers", []) or []
        df = pd.DataFrame({"ticker": [str(t).upper() for t in tickers]})

    watchlist = _load_watchlist()
    if watchlist:
        df = pd.concat([df, pd.DataFrame({"ticker": watchlist})], ignore_index=True)

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    df = df.sort_values("ticker").reset_index(drop=True)
    return df[["ticker"]]


def main() -> None:
    cfg = load_yaml(ROOT / "config.yml") or {}
    df = resolve_universe(cfg)
    df = df[["ticker"]].copy()
    df["asof"] = today_str()

    UNIVERSE.mkdir(parents=True, exist_ok=True)
    df[["ticker"]].to_csv(UNIVERSE / f"universe_{today_str()}.csv", index=False)
    df.to_csv(ROOT / "data" / "universe.csv", index=False)

    con = db_conn()
    con.execute('CREATE TABLE IF NOT EXISTS universe_daily ("asof" DATE, ticker VARCHAR)')
    con.execute("DELETE FROM universe_daily")
    if not df.empty:
        con.register("universe_tmp", df[["asof", "ticker"]])
        con.execute('INSERT INTO universe_daily ("asof", ticker) SELECT "asof", ticker FROM universe_tmp')
        con.unregister("universe_tmp")

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Universe tickers: {len(df)} as of {generated_at}")


if __name__ == "__main__":
    main()
