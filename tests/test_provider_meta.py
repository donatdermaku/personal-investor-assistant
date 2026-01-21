from __future__ import annotations

from pathlib import Path

from src.streamlit_data import CoverageMeta, _read_csv_with_meta, _read_parquet_with_meta, merge_coverage


def test_missing_csv_returns_meta(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    df, meta = _read_csv_with_meta(path, required_cols=["ticker"], tickers=["AAA", "BBB"])
    assert df.empty
    assert isinstance(meta, CoverageMeta)
    assert meta.reasons.get("missing_file") == 1
    assert meta.total == 2
    assert meta.covered == 0


def test_missing_parquet_returns_meta(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    df, meta = _read_parquet_with_meta(path, required_cols=["ticker"], tickers=["AAA"])
    assert df.empty
    assert meta.reasons.get("missing_file") == 1
    assert meta.total == 1
    assert meta.covered == 0


def test_merge_coverage_combines() -> None:
    meta_a = CoverageMeta(total=2, covered=1, missing_tickers=["BBB"], last_date="2024-01-01", reasons={"missing_file": 1}, notes=["a"])
    meta_b = CoverageMeta(total=1, covered=1, missing_tickers=[], last_date="2024-02-01", reasons={"empty_data": 1}, notes=["b"])
    merged = merge_coverage([meta_a, meta_b])
    assert merged.total == 3
    assert merged.covered == 2
    assert merged.last_date == "2024-02-01"
    assert set(merged.missing_tickers) == {"BBB"}
    assert merged.reasons.get("missing_file") == 1
    assert merged.reasons.get("empty_data") == 1
    assert "a" in merged.notes and "b" in merged.notes
