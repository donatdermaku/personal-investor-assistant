import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import pandas as pd
try:
    # duckdb is an optional runtime dependency for some commands (reports may not need it
    # if they only read parquet). Import lazily inside db_conn to avoid top-level ImportError
    # when running parts of the codebase that don't require DuckDB.
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    duckdb = None
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARQ = DATA / "parquet"
DB_PATH = DATA / "db.duckdb"
SEC_CACHE = DATA / "sec"
PARQ.mkdir(parents=True, exist_ok=True)
SEC_CACHE.mkdir(parents=True, exist_ok=True)

def db_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if duckdb is None:
        raise ImportError(
            "duckdb is required for database operations but is not installed or failed to import. "
            "Install it with `pip install duckdb` (or check binary compatibility with your NumPy build)."
        )
    con = duckdb.connect(str(DB_PATH))
    return con

def write_parquet(df: pd.DataFrame, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def today_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_sec_file(url: str, cache_name: str, max_age_hours: int = 24) -> pathlib.Path:
    """Download a lightweight SEC reference file with simple caching."""
    cache_path = SEC_CACHE / cache_name
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return cache_path

    headers = {
        "User-Agent": "personal-investor-assistant (contact: example@example.com)",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


def get_ticker_cik_map(force_refresh: bool = False) -> Dict[str, str]:
    """Return an uppercase ticker -> zero-padded CIK mapping from the SEC file."""
    cache_file = "company_tickers.json"
    if force_refresh:
        cache_path = fetch_sec_file(
            "https://www.sec.gov/files/company_tickers.json", cache_file, max_age_hours=0
        )
    else:
        cache_path = fetch_sec_file(
            "https://www.sec.gov/files/company_tickers.json", cache_file
        )

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }
    return mapping


def register_temp_view(con: Any, name: str, df: pd.DataFrame) -> Optional[str]:
    """Register a pandas DataFrame as a DuckDB view if it has rows."""
    if df is None or df.empty:
        return None
    con.register(name, df)
    return name


def unregister_temp_view(con: Any, name: Optional[str]) -> None:
    if name:
        con.unregister(name)
