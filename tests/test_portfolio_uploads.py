from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.portfolio import load_portfolio

def test_load_portfolio_ignores_files_without_uploads(tmp_path: Path) -> None:
    # Setup Paths
    base_dir = tmp_path / "data" / "user_uploads"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "transactions.csv").write_text("date,ticker,action,quantity,price\n2024-01-01,AAA,BUY,1,100\n")
    (base_dir / "holdings.csv").write_text("ticker,quantity\nAAA,1\n")

    prices = _prices_for("AAA")
    
    # We must patch ROOT because DataManager uses ROOT, not the passed base_dir
    with patch("storage.datamanager.ROOT", tmp_path), \
         patch("storage.datamanager.STORAGE_MODE", "local"):
        result = load_portfolio(prices, ["AAA"], uploads_active=False)

    assert result.daily_values.empty
    assert result.errors
    assert "No uploads" in result.errors[0]


def test_load_portfolio_snapshot_when_uploaded(tmp_path: Path) -> None:
    # Setup Paths
    base_dir = tmp_path / "data" / "user_uploads"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "holdings.csv").write_text("ticker,quantity\nAAA,2\n")

    prices = _prices_for("AAA")
    
    with patch("storage.datamanager.ROOT", tmp_path), \
         patch("storage.datamanager.STORAGE_MODE", "local"):
        result = load_portfolio(prices, ["AAA"], source_override="Snapshot", uploads_active=True)

    assert not result.daily_values.empty
    assert result.source == "snapshot"


def test_load_portfolio_uses_explicit_portfolio_id() -> None:
    prices = _prices_for("AAA")
    captured_ids: list[int] = []

    def _capture_trades(portfolio_id: int) -> pd.DataFrame:
        captured_ids.append(portfolio_id)
        return pd.DataFrame()

    with patch("src.portfolio.data_manager.load_trades", side_effect=_capture_trades), \
         patch("src.portfolio.data_manager.load_snapshot", return_value=pd.DataFrame()):
        result = load_portfolio(
            prices,
            ["AAA"],
            source_override="Ledger",
            uploads_active=True,
            portfolio_id=12345,
        )

    assert captured_ids == [12345]
    assert result.source == "ledger"
    assert result.errors == ["Ledger data empty."]


def _prices_for(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": ticker, "adj_close": 100.0},
            {"date": "2024-01-02", "ticker": ticker, "adj_close": 101.0},
        ]
    )
