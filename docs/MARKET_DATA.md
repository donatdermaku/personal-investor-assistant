# Market Data Backbone

This document defines the market data contract, sources, and caching rules for Nexus Analytics.

## Contracts

### MarketPriceFrame
- Required columns: `date` (YYYY-MM-DD), `close` (float)
- Optional: `adj_close`, `open`, `high`, `low`, `volume`, `ticker`
- Date must be a column (not index), sorted ascending, no duplicates.

### Corporate Actions
- Dividends: `date`, `amount`
- Splits: `date`, `ratio`

### Coverage Summary
Coverage is exported as `coverage_summary.json` and returned by the API to explain
data sufficiency for KPIs. See `docs/COVERAGE_SEMANTICS.md` for structure and rules.

## Sources

Tier 1:
- Yahoo Finance daily prices via `yfinance` (normalized in `market_data/yahoo.py`)

Tier 2:
- FRED series (risk-free rate, CPI) via `market_data/fred.py`

Tier 3:
- Fundamentals via Yahoo Finance `info` (best-effort)

## Caching Rules

Prices:
- Cached per ticker in `data/market_cache/prices/{TICKER}.parquet`
- Refresh if latest date is older than `today - 1 business day`

Dividends/Splits:
- Cached per ticker in `data/market_cache/dividends` and `data/market_cache/splits`

Fundamentals:
- Cached weekly in `data/market_cache/fundamentals`

FRED:
- Cached in `data/market_cache/fred`

## Coverage Alignment

Trade dates are aligned to the nearest prior trading day if a trade falls on a non-trading day.
If a trade date is missing in market data and no prior date exists, an error is raised:
`MARKET_DATA_MISSING_DATES`
