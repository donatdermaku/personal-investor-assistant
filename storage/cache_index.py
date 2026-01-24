from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from storage.repo import use_supabase


@dataclass
class CacheIndexEntry:
    source: str
    key: str
    asof_date: str | None
    updated_at: datetime
    ttl_seconds: int
    status: str
    coverage_pct: float | None
    storage_path: str
    error_code: str | None
    error_message: str | None


def get_cache_entry(source: str, key: str) -> CacheIndexEntry | None:
    if use_supabase():
        from storage_supabase.db import session_scope
        from storage_supabase import models
    else:
        from storage.db import session_scope
        from storage import models

    with session_scope() as session:
        row = (
            session.query(models.DataCacheIndex)
            .filter_by(source=source, key=key)
            .first()
        )
        if not row:
            return None
        return CacheIndexEntry(
            source=row.source,
            key=row.key,
            asof_date=row.asof_date.isoformat() if row.asof_date else None,
            updated_at=row.updated_at,
            ttl_seconds=row.ttl_seconds,
            status=row.status,
            coverage_pct=row.coverage_pct,
            storage_path=row.storage_path,
            error_code=row.error_code,
            error_message=row.error_message,
        )


def upsert_cache_entry(
    *,
    source: str,
    key: str,
    asof_date: str | None,
    ttl_seconds: int,
    status: str,
    coverage_pct: float | None,
    storage_path: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> CacheIndexEntry:
    if use_supabase():
        from storage_supabase.db import session_scope
        from storage_supabase import models
    else:
        from storage.db import session_scope
        from storage import models

    updated_at = datetime.now(timezone.utc)
    parsed_asof = None
    if isinstance(asof_date, str):
        try:
            parsed_asof = datetime.fromisoformat(asof_date)
        except ValueError:
            parsed_asof = None
    elif isinstance(asof_date, datetime):
        parsed_asof = asof_date
    with session_scope() as session:
        row = (
            session.query(models.DataCacheIndex)
            .filter_by(source=source, key=key)
            .first()
        )
        if not row:
            row = models.DataCacheIndex(
                source=source,
                key=key,
            )
            session.add(row)
        row.asof_date = parsed_asof
        row.updated_at = updated_at
        row.ttl_seconds = ttl_seconds
        row.status = status
        row.coverage_pct = coverage_pct
        row.storage_path = storage_path
        row.error_code = error_code
        row.error_message = error_message

    return CacheIndexEntry(
        source=source,
        key=key,
        asof_date=asof_date,
        updated_at=updated_at,
        ttl_seconds=ttl_seconds,
        status=status,
        coverage_pct=coverage_pct,
        storage_path=storage_path,
        error_code=error_code,
        error_message=error_message,
    )
