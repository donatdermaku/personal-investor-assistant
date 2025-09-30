from datetime import datetime
import pathlib
import sys

import pandas as pd
import yfinance as yf

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils_io import (
    PARQ,
    ROOT,
    db_conn,
    load_yaml,
    register_temp_view,
    today_str,
    unregister_temp_view,
)

def main():
    cfg = load_yaml(ROOT / "watchlist.yml")
    tickers = cfg["tickers"]
    start = "2015-01-01"  # initial backfill
    end = datetime.utcnow().strftime("%Y-%m-%d")

    data = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            continue
        df = df.reset_index().rename(columns=str.lower)
        df["ticker"] = t
        df.rename(columns={"adj close":"adj_close"}, inplace=True)
        data.append(df[["date","ticker","open","high","low","close","adj_close","volume"]])

    prices = pd.concat(data, ignore_index=True) if data else pd.DataFrame(
        columns=["date","ticker","open","high","low","close","adj_close","volume"]
    )

    # Write to DuckDB + Parquet snapshot
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
        con.execute(f"INSERT INTO prices_daily SELECT * FROM {view_name}")
    unregister_temp_view(con, view_name)

    from src.utils_io import write_parquet
    write_parquet(prices, PARQ / f"prices_daily_{today_str()}.parquet")
    print(f"Saved prices for {len(tickers)} tickers.")

if __name__ == "__main__":
    main()
