from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import os

import pandas as pd

from src.utils_io import ROOT
from storage.cache_index import CacheIndexEntry, get_cache_entry, upsert_cache_entry
from storage.repo import use_supabase


@dataclass
class CacheResult:
    frame: pd.DataFrame
    status: str
    entry: CacheIndexEntry | None


def _cache_root() -> Path:
    override = os.getenv("NEXUS_MARKET_CACHE_DIR")
    if override:
        return Path(override)
    return ROOT / "data" / "market_cache" / "persistent"


def _local_cache_path(source: str, key: str) -> Path:
    base = _cache_root() / source
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{key}.parquet"


def _storage_path(source: str, key: str) -> str:
    return f"cache/{source}/{key}.parquet"


def _read_parquet_bytes(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(BytesIO(data))
    except Exception:
        return pd.DataFrame()


def _write_parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def load_cached_frame(source: str, key: str) -> CacheResult:
    entry = get_cache_entry(source, key)
    local_path = _local_cache_path(source, key)
    if local_path.exists():
        try:
            return CacheResult(pd.read_parquet(local_path), entry.status if entry else "fresh", entry)
        except Exception:
            pass
    if use_supabase():
        from storage_supabase.storage import download_bytes
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "nexus-artifacts")
        try:
            data = download_bytes(bucket, _storage_path(source, key))
            frame = _read_parquet_bytes(data)
            if not frame.empty:
                frame.to_parquet(local_path, index=False)
                if entry is None:
                    entry = upsert_cache_entry(
                        source=source,
                        key=key,
                        asof_date=None,
                        ttl_seconds=0,
                        status="stale",
                        coverage_pct=None,
                        storage_path=_storage_path(source, key),
                    )
                return CacheResult(frame, entry.status if entry else "stale", entry)
        except Exception:
            return CacheResult(pd.DataFrame(), "error", entry)
    return CacheResult(pd.DataFrame(), "error", entry)


def store_cached_frame(
    *,
    source: str,
    key: str,
    frame: pd.DataFrame,
    ttl_seconds: int,
    asof_date: str | None,
    coverage_pct: float | None,
    status: str = "fresh",
    error_code: str | None = None,
    error_message: str | None = None,
) -> CacheIndexEntry:
    local_path = _local_cache_path(source, key)
    frame.to_parquet(local_path, index=False)
    storage_path = _storage_path(source, key)
    if use_supabase():
        from storage_supabase.storage import upload_bytes
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "nexus-artifacts")
        upload_bytes(bucket, storage_path, _write_parquet_bytes(frame), "application/octet-stream")
    return upsert_cache_entry(
        source=source,
        key=key,
        asof_date=asof_date,
        ttl_seconds=ttl_seconds,
        status=status,
        coverage_pct=coverage_pct,
        storage_path=storage_path,
        error_code=error_code,
        error_message=error_message,
    )


def get_or_refresh_frame(
    *,
    source: str,
    key: str,
    ttl_seconds: int,
    fetch_fn,
    asof_date: str | None = None,
    allow_refresh: bool = True,
    force_refresh: bool = False,
) -> CacheResult:
    entry = get_cache_entry(source, key)
    result = load_cached_frame(source, key)
    now = datetime.now(timezone.utc)
    stale = False
    if entry and entry.updated_at and entry.ttl_seconds:
        updated_at = entry.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = (now - updated_at).total_seconds()
        stale = age > entry.ttl_seconds
        if not force_refresh and age <= entry.ttl_seconds and not result.frame.empty:
            return CacheResult(result.frame, "fresh", entry)
    if not allow_refresh:
        status = "stale" if stale else (entry.status if entry else "error")
        return CacheResult(result.frame, status, entry)
    try:
        fetched = fetch_fn()
        if fetched is None or fetched.empty:
            raise ValueError("Fetched frame is empty.")
        updated = store_cached_frame(
            source=source,
            key=key,
            frame=fetched,
            ttl_seconds=ttl_seconds,
            asof_date=asof_date,
            coverage_pct=1.0,
            status="fresh",
        )
        return CacheResult(fetched, "fresh", updated)
    except Exception as exc:
        if not result.frame.empty:
            updated = store_cached_frame(
                source=source,
                key=key,
                frame=result.frame,
                ttl_seconds=ttl_seconds,
                asof_date=asof_date,
                coverage_pct=entry.coverage_pct if entry else None,
                status="stale",
                error_code="CACHE_REFRESH_FAILED",
                error_message=str(exc),
            )
            return CacheResult(result.frame, "stale", updated)
        return CacheResult(pd.DataFrame(), "error", entry)
