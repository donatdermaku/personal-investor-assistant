#!/usr/bin/env python3
"""Scan data/parquet for parquet files that fail to read and move them to data/parquet/_corrupt.
Print commands to regenerate them via the ingestion scripts.

Usage:
    python scripts/repair_parquet.py --move

If --move is omitted, the script only reports files that fail to read.
"""
import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARQ = ROOT / "data" / "parquet"
CORRUPT = PARQ / "_corrupt"


def check_file(p: pathlib.Path) -> bool:
    try:
        pd.read_parquet(p)
        return True
    except Exception as e:
        print(f"BAD: {p} -> {e}")
        return False


def main(move: bool):
    files = sorted(PARQ.glob("*.parquet"))
    if not files:
        print("No parquet files found in data/parquet")
        return
    bad = []
    for f in files:
        ok = check_file(f)
        if not ok:
            bad.append(f)
    if not bad:
        print("All parquet files readable")
        return
    print("\nCorrupted parquet files:")
    for b in bad:
        print(b)
    if move:
        CORRUPT.mkdir(parents=True, exist_ok=True)
        for b in bad:
            dest = CORRUPT / b.name
            b.rename(dest)
            print(f"Moved {b} -> {dest}")
    print("\nTo regenerate snapshots, run:")
    print("python -m src.ingest_prices")
    print("python -m src.ingest_fundamentals_sec")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--move", action="store_true", help="Move corrupted files to data/parquet/_corrupt")
    args = p.parse_args()
    main(args.move)
