# Personal Investor Assistant — Watchlist Pulse

Static, universe-first investor analytics built on SEC EDGAR + yfinance, rendered as a GitHub Pages dashboard.

## What it does
- Builds a scoring universe (S&P 500 / Nasdaq 100 / manual list) and computes z-score factor analytics against that universe.
- Produces a daily dashboard in `reports/index.html` with sortable/filterable watchlist tables.
- Generates a daily pulse summary (`reports/pulse.json`, `reports/pulse.md`) and a pulse archive.
- Creates per-ticker drilldown pages for watchlist tickers.

## Key folders
- `src/ingest/`: universe, price, and SEC fundamentals ingestion
- `src/compute/`: factor modeling + QA checks
- `src/report/`: HTML/JSON/Markdown outputs
- `data/`: DuckDB + cached JSON + Parquet snapshots
- `reports/`: GitHub Pages output

## Config
Edit `config.yml`:
```yaml
universe:
  mode: "sp500"   # manual | sp500 | nasdaq100
  tickers: []
  min_size: 200

weights:
  value: 0.4
  quality: 0.4
  momentum: 0.2

report:
  ticker_history_days: 365
  max_history_snapshots: 260
```

Watchlist lives in `watchlist.yml`.

## Run locally
```bash
python -m src.ingest_universe
python -m src.ingest_prices
python -m src.ingest_fundamentals_sec
python -m src.compute_factors
python -m src.build_report
```

## GitHub Actions
The nightly workflow runs the same sequence and uploads `reports/` as the Pages artifact. Set the `SEC_USER_AGENT` secret for EDGAR compliance.

## Extend data sources
- Add more universe providers in `src/ingest/universe.py`.
- Add new factor definitions in `src/compute/factors.py`.
- Adjust pulse logic in `src/report/build.py`.

## Notes
- No Parquet or report artifacts are committed to the repo.
- This project is for educational use only.
