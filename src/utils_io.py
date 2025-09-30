import json, os, sys, time, pathlib
from typing import Any, Dict, Iterable
from datetime import datetime
import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARQ = DATA / "parquet"
DB_PATH = DATA / "db.duckdb"
PARQ.mkdir(parents=True, exist_ok=True)

def db_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    return con

def write_parquet(df: pd.DataFrame, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
