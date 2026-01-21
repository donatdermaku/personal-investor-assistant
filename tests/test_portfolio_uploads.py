from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.portfolio import load_portfolio


def _prices_for(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": ticker, "adj_close": 100.0},
            {"date": "2024-01-02", "ticker": ticker, "adj_close": 101.0},
        ]
    )


def test_load_portfolio_ignores_files_without_uploads(tmp_path: Path) -> None:
    base_dir = tmp_path
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "transactions.csv").write_text("date,ticker,action,quantity,price\n2024-01-01,AAA,BUY,1,100\n")
    (base_dir / "holdings.csv").write_text("ticker,quantity\nAAA,1\n")

    prices = _prices_for("AAA")
    result = load_portfolio(prices, ["AAA"], uploads_active=False, base_dir=base_dir)

    assert result.daily_values.empty
    assert result.errors
    assert "No uploads" in result.errors[0]


def test_load_portfolio_snapshot_when_uploaded(tmp_path: Path) -> None:
    base_dir = tmp_path
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "holdings.csv").write_text("ticker,quantity\nAAA,2\n")

    prices = _prices_for("AAA")
    result = load_portfolio(prices, ["AAA"], source_override="Snapshot", uploads_active=True, base_dir=base_dir)

    assert not result.daily_values.empty
    assert result.source == "snapshot"
