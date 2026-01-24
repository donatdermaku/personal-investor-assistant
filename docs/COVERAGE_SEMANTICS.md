# Coverage Semantics

Nexus expresses data coverage as a structured, quantitative object. Coverage is **not** a boolean.
The summary is exported as `coverage_summary.json` and returned by the API.

## Coverage Summary Shape

```json
{
  "as_of": "YYYY-MM-DD",
  "status": "sufficient | insufficient | unknown",
  "score": 0.0,
  "policy": {
    "min_score_for_kpis": 0.95,
    "min_history_days": 252,
    "max_gap_days": 5
  },
  "required": {
    "tickers": ["AAPL", "MSFT", "SPY"],
    "history_days_needed": 252
  },
  "per_ticker": {
    "AAPL": {
      "score": 1.0,
      "history_days": 1200,
      "missing_days": 0,
      "largest_gap_days": 0,
      "status": "ok",
      "reason_codes": []
    }
  },
  "aggregate": {
    "coverage_ratio": 0.98,
    "min_ticker_score": 0.96,
    "benchmark_score": 1.0,
    "rf_score": null
  },
  "reason_codes": ["OK"],
  "contract_version": "coverage_summary_v1"
}
```

## Scoring Rules

- **History sufficiency**: `min(history_days / required_days, 1)`
- **Missingness penalty**: `1 - (missing_days / required_days)`
- **Gap penalty**: no penalty for gaps up to `max_gap_days`, linear penalty beyond.
- **Score**: `history_score * missing_penalty * gap_penalty`

## Status Rules

- `unknown`: coverage not computed or data unavailable.
- `insufficient`: score below `min_score_for_kpis` or core series missing.
- `sufficient`: score meets or exceeds `min_score_for_kpis`.

Risk-free rate coverage is tracked separately as `aggregate.rf_score` and may add
`RF_MISSING` to `reason_codes` without changing the overall status.

## UI Semantics

- `unknown`: KPI values are shown, no warning badge.
- `insufficient`: KPI values are hidden and a reason code is shown.
- `sufficient`: KPI values are shown with coverage badge.
