from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.prices import _split_multiindex


def _legacy_split_multiindex(df: pd.DataFrame, tickers: list[str], vendor_map: dict[str, str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if df.empty:
        return frames
    if isinstance(df.columns, pd.MultiIndex):
        level = df.columns.get_level_values(1)
        for vendor in vendor_map.keys():
            if vendor in level:
                sub = df.xs(vendor, axis=1, level=1)
                frames.append(_normalize_like_prod(sub, vendor_map[vendor]))
        return frames
    if tickers:
        frames.append(_normalize_like_prod(df, vendor_map.get(tickers[0], tickers[0])))
    return frames


def _normalize_like_prod(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.reset_index().copy()
    out.columns = [str(c).lower() for c in out.columns]
    out["ticker"] = ticker
    if "adj close" in out.columns:
        out = out.rename(columns={"adj close": "adj_close"})
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def test_split_multiindex_matches_legacy_for_ten_tickers() -> None:
    tickers = [f"T{i:02d}" for i in range(10)]
    vendor_map = {t: t for t in tickers}
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    cols = pd.MultiIndex.from_product([fields, tickers])
    arr = np.zeros((len(dates), len(cols)), dtype=float)
    for j, (_, ticker) in enumerate(cols):
        base = 100 + j
        if fields[j // len(tickers)] == "Volume":
            arr[:, j] = np.arange(len(dates)) + 1000 + j
        else:
            arr[:, j] = base + np.linspace(0, 1, len(dates))

    raw = pd.DataFrame(arr, index=dates, columns=cols)
    raw.index.name = "Date"

    new_frames = _split_multiindex(raw, tickers, vendor_map)
    old_frames = _legacy_split_multiindex(raw, tickers, vendor_map)

    assert len(new_frames) == len(old_frames) == 10

    old_by_ticker = {f["ticker"].iloc[0]: f.sort_values("date").reset_index(drop=True) for f in old_frames}
    new_by_ticker = {f["ticker"].iloc[0]: f.sort_values("date").reset_index(drop=True) for f in new_frames}

    for ticker in tickers:
        lhs = new_by_ticker[ticker]
        rhs = old_by_ticker[ticker]
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            assert np.allclose(lhs[col].to_numpy(dtype=float), rhs[col].to_numpy(dtype=float), atol=1e-12)
