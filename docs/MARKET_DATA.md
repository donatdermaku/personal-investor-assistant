# Market Data Backbone

This document defines the market data contract, sources, and caching rules for Nexus Analytics.

## Contracts

### MarketPriceFrame
- Required columns: `date` (YYYY-MM-DD), `close` (float)
- Optional: `adj_close`, `open`, `high`, `low`, `volume`, `ticker`
- Date must be a column (not index), sorted ascending, no duplicates.
 - Corporate actions normalized: `dividend` (cash per share) and `split_ratio` (default 1.0).

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

## Exports

- `coverage_summary.json` (coverage semantics)
- `risk_free_series.csv` (DTB3 aligned to portfolio calendar)
- `corporate_actions_events.csv` (dividends and splits)

## Coverage Alignment

Trade dates are aligned to the nearest prior trading day if a trade falls on a non-trading day.
If a trade date is missing in market data and no prior date exists, an error is raised:
`MARKET_DATA_MISSING_DATES`

## Risk-Free Rate

Default series: **DTB3** (3-month T-bill). The daily risk-free return is derived as:
`(1 + annual_rate)^(1/252) - 1`, aligned to the portfolio return calendar.
If the series is missing, Sharpe uses a zero risk-free rate and coverage flags the context as missing.

## Calendar Alignment

- **Comparative metrics** (tracking error, correlation, active returns): use the **intersection**
  of portfolio and benchmark calendars.
- **Portfolio valuation** (equity curve): uses the portfolio valuation calendar; gaps are
  tolerated with explicit assumptions.
