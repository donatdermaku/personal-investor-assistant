import os
from pathlib import Path
from typing import Any
from datetime import datetime
import pandas as pd

from storage import repo, db
from src.utils_io import ROOT

# Config
# modes: "db", "files", "hybrid"
STORAGE_MODE = os.getenv("STORAGE_MODE", "hybrid")

class DataManager:
    """
    Facade for all data operations.
    Handles 'User Data' (SQLite vs Files) and 'Market Data' (DuckDB).
    """
    def __init__(self):
        # Initialize DB if needed
        if STORAGE_MODE in ("db", "hybrid"):
            db.init_db()

    # --- User Identity ---
    def get_current_user_id(self) -> int:
        if STORAGE_MODE == "files":
            return 0 # Mock ID
        return repo.get_user_id()

    def get_main_portfolio_id(self, user_id: int) -> int:
        if STORAGE_MODE == "files":
            return 0
        return repo.get_default_portfolio_id(user_id)

    # --- Watchlist ---
    def load_watchlist(self) -> list[str]:
        if STORAGE_MODE == "files":
            return self._load_watchlist_file()
        
        # Hybrid/DB: Try DB first
        user_id = self.get_current_user_id()
        tickers = repo.list_watch_tickers(user_id)
        
        if not tickers and STORAGE_MODE == "hybrid":
            # Fallback to file if DB empty
            file_tickers = self._load_watchlist_file()
            if file_tickers:
                # Optional: Auto-migrate to DB on read? 
                # For now, just return file data, or better: keep clean separation.
                # Let's return file data to respect "Hybrid means fallback".
                return file_tickers
        
        return tickers

    def save_watchlist(self, tickers: list[str]):
        if STORAGE_MODE in ("db", "hybrid"):
            user_id = self.get_current_user_id()
            repo.replace_watchlist(user_id, tickers)
        
        if STORAGE_MODE in ("files", "hybrid"):
            self._save_watchlist_file(tickers)

    # --- File/Legacy Helpers ---
    def _load_watchlist_file(self) -> list[str]:
        import yaml
        path = ROOT / "watchlist.yml"
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text())
            return data.get("tickers", [])
        except Exception:
            return []

    def _save_watchlist_file(self, tickers: list[str]):
        import yaml
        path = ROOT / "watchlist.yml"
        data = {"tickers": tickers}
        with open(path, "w") as f:
            yaml.dump(data, f)

    # --- Portfolio Inputs ---
    def load_trades(self, portfolio_id: int) -> pd.DataFrame:
        if STORAGE_MODE == "files":
            return self._load_trades_file()
        
        data = repo.get_trades(portfolio_id)
        if not data and STORAGE_MODE == "hybrid":
             return self._load_trades_file()
        
        if not data:
            return pd.DataFrame()
        
        # Convert list of dicts to DF and cleanup cols
        df = pd.DataFrame(data)
        # Rename 'shares'->'quantity' to match portfolio.py expectation if needed?
        # portfolio.py expects "quantity". Schema said "shares". 
        # let's map it here or in repo.
        if "shares" in df.columns:
            df = df.rename(columns={"shares": "quantity"})
        return df

    def load_snapshot(self, portfolio_id: int) -> pd.DataFrame:
        if STORAGE_MODE == "files":
            return self._load_snapshot_file()
        
        data = repo.get_latest_snapshot(portfolio_id)
        if not data and STORAGE_MODE == "hybrid":
            return self._load_snapshot_file()
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if "shares" in df.columns:
            df = df.rename(columns={"shares": "quantity"})
        return df

    def _load_trades_file(self) -> pd.DataFrame:
        path = ROOT / "data" / "user_uploads" / "transactions.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _load_snapshot_file(self) -> pd.DataFrame:
        path = ROOT / "data" / "user_uploads" / "holdings.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def save_portfolio_inputs(self, portfolio_id: int, trades: pd.DataFrame | None, snapshot: pd.DataFrame | None):
        """
        Save inputs to DB (and file if hybrid). 
        Accepts DataFrames with UI column names.
        """
        # Save Trades
        if trades is not None:
            # Map DF to list of dicts
            # "quantity" -> "shares"
            t_df = trades.copy()
            if "quantity" in t_df.columns:
                t_df = t_df.rename(columns={"quantity": "shares"})
            
            # Fill missing
            required = ["date", "ticker", "action", "shares", "amount"]
            for r in required:
                if r not in t_df.columns:
                    t_df[r] = 0 if r in ["shares", "amount"] else ""
            
            # Ensure date format
            t_df["date"] = pd.to_datetime(t_df["date"])
            
            if STORAGE_MODE in ("db", "hybrid"):
                records = t_df.to_dict(orient="records")
                repo.replace_trades(portfolio_id, records)
            
            if STORAGE_MODE in ("files", "hybrid"):
                path = ROOT / "data" / "user_uploads" / "transactions.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                # Save as CSV expected by legacy
                # legacy expects 'quantity'
                trades.to_csv(path, index=False)

        # Save Snapshot
        if snapshot is not None:
            s_df = snapshot.copy()
            if "quantity" in s_df.columns:
                s_df = s_df.rename(columns={"quantity": "shares"})
            
            if "cost_basis" not in s_df.columns:
                s_df["cost_basis"] = 0.0
                
            s_df["as_of_date"] = datetime.utcnow() # Snapshots usually implied 'now'
            
            if STORAGE_MODE in ("db", "hybrid"):
                records = s_df.to_dict(orient="records")
                repo.replace_snapshot(portfolio_id, records)
                
            if STORAGE_MODE in ("files", "hybrid"):
                path = ROOT / "data" / "user_uploads" / "holdings.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                snapshot.to_csv(path, index=False)

# Global Singleton
data_manager = DataManager()
