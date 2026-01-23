# Export Specification

## Overview
All exports must be derived from a consistent `AppState` snapshot to ensure that the UI and the exported files match exactly.

Exports include:
1.  `summary.json` (Machine-readable run metadata & core metrics)
2.  `report.html` (Human-readable, self-contained interactive report)
3.  `data/*.csv` (Raw data files for external analysis)

## 1. Summary JSON Schema (`summary.json`)
```json
{
  "run_id": "uuid-string",
  "timestamp": "ISO-8601-UTC",
  "input_hash": "sha256-hash",
  "data_hash": "sha256-hash",
  "code_version": "string",
  "metrics": {
    "twr": float | null,
    "mwr": float | null,
    "final_value": float,
    "total_return_pct": float,
    "max_drawdown": float,
    "sharpe_ratio": float | null
  },
  "coverage": {
    "prices": "N/M",
    "fundamentals": "N/M"
  },
  "errors": ["list", "of", "strings"]
}
```

## 2. HTML Report Structure (`report.html`)
The HTML report is a single-file, offline-capable document.

**Sections:**
1.  **Header**: 
    - Title: "Personal Investor Assistant Report"
    - Metadata: Run ID, Date, Input Hash (small print)
2.  **Executive Summary**:
    - Key Metrics Grid: TWR, MWR, Net Worth, Drawdown
3.  **Equity Curve**:
    - Interactive Plotly chart (embedded JSON/JS or base64 image)
4.  **Portfolio Composition**:
    - Top Holdings Table
    - Allocation Pie Chart
5.  **Risk Profile**:
    - Rolling Volatility / Sharpe
    - Component Risk Table
6.  **Data Quality Appendix**:
    - Missing Tickers List
    - Coverage Definitions
    - Glossary (TWR vs MWR, etc.)

**Technical Constraints:**
- Must use system fonts (Inter/Roboto fallback).
- CSS should be embedded in `<style>` block.
- JavaScript (Plotly) should be loaded via CDN with graceful fallback, or embedded if feasible without bloating size >5MB.

## 3. CSV Exports
- `portfolio_daily.csv`: [date, value, cashflow, return]
- `transactions_clean.csv`: [date, ticker, amount, type]
- `holdings.csv`: [ticker, shares, weight, value]
