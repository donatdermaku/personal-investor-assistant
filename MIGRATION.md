# Migration Guide

## What changed
- Universe-first scoring: z-scores are computed against the universe, not the watchlist.
- New outputs: pulse JSON/MD, per-ticker drilldowns, assets folder.
- New modules: `src/ingest`, `src/compute`, `src/report` (wrappers remain for CLI).

## What to delete / keep
- Keep: `data/` cache and `data/parquet/` snapshots if you want continuity.
- Optional: delete `data/` to force a full rebuild of universe, prices, fundamentals.
- Keep `watchlist.yml` and `config.yml` (updated with universe settings).

## Breaking changes
- `config.yml` now includes `universe` settings.
- `reports/` now contains `pulse/` archive, `assets/`, and per-ticker pages.
- `scores_daily` table schema expanded.

## Data reset guidance
If factors or universe config change materially, delete:
- `data/parquet/scores_daily_*.parquet`
- `data/db.duckdb`
Then rerun the pipeline.
