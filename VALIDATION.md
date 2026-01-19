# Validation Checklist

- Universe ingestion produces `data/universe.csv` with >= 200 tickers (or emits robust warning).
- Prices and fundamentals exist in DuckDB and latest Parquet snapshots.
- `scores_daily_YYYY-MM-DD.parquet` exists and `Composite` is not all NaN/0.
- `reports/index.html` exists and references `assets/report.css`.
- `reports/pulse.json` and `reports/pulse.md` exist.
- `reports/ticker/<TICKER>.html` exists for each watchlist ticker.
