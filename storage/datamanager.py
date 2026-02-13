import os
import logging
from datetime import datetime
from typing import Any
import pandas as pd

from storage import repo, db
from src.utils_io import ROOT

_LOGGER = logging.getLogger(__name__)
_VALID_STORAGE_MODES = {"local", "supabase"}
_LEGACY_MODE_MAP = {
    "files": "local",
    "db": "supabase",
    "hybrid": "supabase",
}


def _resolve_storage_mode(raw_mode: str | None) -> str:
    normalized = (raw_mode or "local").strip().lower()
    if normalized in _LEGACY_MODE_MAP:
        mapped = _LEGACY_MODE_MAP[normalized]
        _LOGGER.warning(
            "Legacy STORAGE_MODE=%r detected; using STORAGE_MODE=%r.",
            normalized,
            mapped,
        )
        return mapped
    if normalized not in _VALID_STORAGE_MODES:
        _LOGGER.warning(
            "Invalid STORAGE_MODE=%r detected; defaulting to STORAGE_MODE='local'.",
            normalized,
        )
        return "local"
    return normalized


STORAGE_MODE = _resolve_storage_mode(os.getenv("STORAGE_MODE", "local"))

class DataManager:
    """
    Facade for user data operations.
    - local: file-backed uploads/watchlist for local development and tests
    - supabase: repository-backed storage (local sqlite backend or Supabase backend)
    """
    def __init__(self):
        # Initialize DB when using repository-backed mode.
        if self._uses_repo_backend():
            db.init_db()

    def _uses_local_files(self) -> bool:
        return STORAGE_MODE == "local"

    def _uses_repo_backend(self) -> bool:
        return STORAGE_MODE == "supabase"

    # --- User Identity ---
    def get_current_user_id(self) -> Any:
        if self._uses_local_files():
            return 0  # Mock ID
        return repo.get_user_id()

    def get_main_portfolio_id(self, user_id: int | str) -> int:
        if self._uses_local_files():
            return 0
        return repo.get_default_portfolio_id(user_id)

    # --- Watchlist ---
    def load_watchlist(self) -> list[str]:
        if self._uses_local_files():
            return self._load_watchlist_file()

        user_id = self.get_current_user_id()
        return repo.list_watch_tickers(user_id)

    def save_watchlist(self, tickers: list[str]):
        if self._uses_repo_backend():
            user_id = self.get_current_user_id()
            repo.replace_watchlist(user_id, tickers)
            return

        if self._uses_local_files():
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
        if self._uses_local_files():
            return self._load_trades_file()

        data = repo.get_trades(portfolio_id)
        if not data:
            return pd.DataFrame()

        # Convert list of dicts to DF and cleanup cols
        df = pd.DataFrame(data)
        if "shares" in df.columns:
            df = df.rename(columns={"shares": "quantity"})
        return df

    def load_snapshot(self, portfolio_id: int) -> pd.DataFrame:
        if self._uses_local_files():
            return self._load_snapshot_file()

        data = repo.get_latest_snapshot(portfolio_id)
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
        """Save inputs to the backend selected by STORAGE_MODE."""
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

            if self._uses_repo_backend():
                records = t_df.to_dict(orient="records")
                repo.replace_trades(portfolio_id, records)
            elif self._uses_local_files():
                path = ROOT / "data" / "user_uploads" / "transactions.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                trades.to_csv(path, index=False)

        # Save Snapshot
        if snapshot is not None:
            s_df = snapshot.copy()
            if "quantity" in s_df.columns:
                s_df = s_df.rename(columns={"quantity": "shares"})
            
            if "cost_basis" not in s_df.columns:
                s_df["cost_basis"] = 0.0
                
            s_df["as_of_date"] = datetime.utcnow()

            if self._uses_repo_backend():
                records = s_df.to_dict(orient="records")
                repo.replace_snapshot(portfolio_id, records)
            elif self._uses_local_files():
                path = ROOT / "data" / "user_uploads" / "holdings.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                snapshot.to_csv(path, index=False)

# Global Singleton
data_manager = DataManager()
