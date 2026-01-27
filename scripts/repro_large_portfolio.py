from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["STORAGE_MODE"] = "files"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from storage.datamanager import data_manager  # noqa: E402
from src.pipeline import compute_app_state, save_artifacts  # noqa: E402
from src.utils_io import ROOT  # noqa: E402
from src.utils_memory import log_rss  # noqa: E402


def main() -> None:
    trades_path = ROOT / "large_portfolio_trades_contract_v1_bmonthend.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades file: {trades_path}")

    trades = pd.read_csv(trades_path)
    tickers = sorted(set(trades["ticker"].dropna().astype(str).str.upper()))
    if "CASH" in tickers:
        tickers.remove("CASH")

    data_manager.save_watchlist(tickers)
    data_manager.save_portfolio_inputs(0, trades=trades, snapshot=None)

    log_rss("before_compute")
    app_state = compute_app_state(save_run=False, run_type="repro-large")
    save_artifacts(app_state)
    log_rss("after_compute")


if __name__ == "__main__":
    main()
