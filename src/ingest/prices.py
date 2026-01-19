from __future__ import annotations

from datetime import datetime, timezone
import pathlib
import sys
import time

import pandas as pd
import yfinance as yf

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.utils_io import (
    PARQ,
    ROOT,
    db_conn,
    load_yaml,
    register_temp_view,
    today_str,
    unregister_temp_view,
)
from src.utils_io import normalize_ticker, yahoo_ticker


def _load_universe_tickers() -> pd.DataFrame:
    uni_path = ROOT / "data" / "universe.csv"
    if uni_path.exists():
        df = pd.read_csv(uni_path)
        if "vendor_ticker" not in df.columns:
            df["vendor_ticker"] = df["ticker"].apply(yahoo_ticker)
        df["ticker"] = df["ticker"].apply(normalize_ticker)
        df["vendor_ticker"] = df["vendor_ticker"].apply(yahoo_ticker)
        return df[["ticker", "vendor_ticker"]]
    watch = load_yaml(ROOT / "watchlist.yml") or {}
    tickers = [normalize_ticker(t) for t in watch.get("tickers", [])]
    return pd.DataFrame({"ticker": tickers, "vendor_ticker": [yahoo_ticker(t) for t in tickers]})


def _download_chunk(tickers: list[str], start: str, end: str, retries: int, backoff: int) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(
                tickers=" ".join(tickers),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            return df
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[WARN] Price download failed for chunk {tickers}: {exc}")
                return pd.DataFrame()
            time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()


def _normalize_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.reset_index()
    cols = [str(c).lower() for c in df.columns]
    df.columns = cols
    df["ticker"] = ticker
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def _split_multiindex(df: pd.DataFrame, tickers: list[str], vendor_map: dict[str, str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if df.empty:
        return frames
    if isinstance(df.columns, pd.MultiIndex):
        level = df.columns.get_level_values(1)
        for vendor in vendor_map.keys():
            if vendor in level:
                sub = df.xs(vendor, axis=1, level=1)
                frames.append(_normalize_price_frame(sub, vendor_map[vendor]))
        return frames
    # Single ticker
    if tickers:
        frames.append(_normalize_price_frame(df, vendor_map.get(tickers[0], tickers[0])))
    return frames


def main() -> None:
    cfg = load_yaml(ROOT / "config.yml") or {}
    fetch_cfg = cfg.get("fetch", {})
    retries = int(fetch_cfg.get("retries", 3))
    backoff = int(fetch_cfg.get("backoff_seconds", 2))

    universe = _load_universe_tickers()
    tickers = universe["ticker"].tolist()
    vendor_tickers = universe["vendor_ticker"].tolist()
    vendor_map = dict(zip(vendor_tickers, tickers))
    start = "2015-01-01"
    end = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    data: list[pd.DataFrame] = []
    chunk_size = 50
    for i in range(0, len(vendor_tickers), chunk_size):
        chunk = vendor_tickers[i : i + chunk_size]
        raw = _download_chunk(chunk, start, end, retries, backoff)
        if raw.empty:
            for miss in chunk:
                print(f"[WARN] Missing price data for vendor ticker {miss} (mapped to {vendor_map.get(miss)})")
            continue
        data.extend(_split_multiindex(raw, chunk, vendor_map))
        returned = set()
        if isinstance(raw.columns, pd.MultiIndex):
            returned = set(raw.columns.get_level_values(1))
        elif not raw.empty and chunk:
            returned = {chunk[0]}
        for miss in [t for t in chunk if t not in returned]:
            print(f"[WARN] Missing price data for vendor ticker {miss} (mapped to {vendor_map.get(miss)})")

    prices = pd.concat(data, ignore_index=True) if data else pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    )

    con = db_conn()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_daily (
            date DATE, ticker VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, adj_close DOUBLE, volume BIGINT
        )
        """
    )
    if tickers:
        con.execute(
            "DELETE FROM prices_daily WHERE ticker IN ({})".format(
                ",".join(["?"] * len(tickers))
            ),
            tickers,
        )
    view_name = register_temp_view(con, "prices_tmp", prices)
    if view_name:
        con.execute(
            f"INSERT INTO prices_daily (date, ticker, open, high, low, close, adj_close, volume) "
            f"SELECT date, ticker, open, high, low, close, adj_close, volume FROM {view_name}"
        )
    unregister_temp_view(con, view_name)

    from src.utils_io import write_parquet

    write_parquet(prices, PARQ / f"prices_daily_{today_str()}.parquet")
    print(f"Saved prices for {len(tickers)} tickers.")


if __name__ == "__main__":
    main()
