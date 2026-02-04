"""
Coverage metadata utilities.

Extracted from streamlit_data.py to remove streamlit dependency.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CoverageMeta:
    """Metadata about data coverage for a dataset."""
    total: int
    covered: int
    missing_tickers: list[str]
    last_date: str | None
    reasons: dict[str, int]
    notes: list[str]


def merge_coverage(metas: list[CoverageMeta]) -> CoverageMeta:
    """Merge multiple coverage metadata objects."""
    total = sum(meta.total for meta in metas)
    covered = sum(meta.covered for meta in metas)
    missing = sorted({t for meta in metas for t in meta.missing_tickers})
    dates = [meta.last_date for meta in metas if meta.last_date]
    last_date = max(dates) if dates else None
    reasons: dict[str, int] = {}
    notes: list[str] = []
    for meta in metas:
        for key, value in meta.reasons.items():
            reasons[key] = reasons.get(key, 0) + value
        notes.extend(meta.notes)
    return CoverageMeta(total, covered, missing, last_date, reasons, notes)
