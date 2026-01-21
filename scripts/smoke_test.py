from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def latest_parquet(prefix: str) -> pathlib.Path | None:
    files = sorted((ROOT / "data" / "parquet").glob(f"{prefix}_*.parquet"))
    return files[-1] if files else None


def main() -> int:
    failures = []

    universe_path = ROOT / "data" / "universe.csv"
    if not universe_path.exists():
        failures.append("Missing data/universe.csv")
    else:
        df = pd.read_csv(universe_path)
        if df.empty:
            failures.append("Universe is empty")

    prices_path = latest_parquet("prices_daily")
    fundamentals_path = latest_parquet("fundamentals_quarterly")
    scores_path = latest_parquet("scores_daily")

    if not prices_path:
        failures.append("Missing prices parquet")
    if not fundamentals_path:
        failures.append("Missing fundamentals parquet")
    if not scores_path:
        failures.append("Missing scores parquet")
    else:
        scores = pd.read_parquet(scores_path)
        if scores.empty:
            failures.append("Scores parquet empty")
        else:
            comp = scores.get("Composite")
            if comp is None or comp.isna().all() or (comp.fillna(0) == 0).all():
                failures.append("Composite all NaN/0")

    report_path = ROOT / "reports" / "index.html"
    if not report_path.exists():
        failures.append("Missing reports/index.html")

    if failures:
        print("Smoke test failed:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
