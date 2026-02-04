# Input Contract: Trades CSV Format

## Overview

The Personal Investor Assistant accepts trade ledger data in CSV format. This document specifies the exact format requirements.

## Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | Date (YYYY-MM-DD) | Trade execution date | `2023-01-03` |
| `ticker` | String | Stock ticker symbol (uppercase) | `AAPL` |
| `action` | String | Trade action type | `BUY`, `SELL`, `DEPOSIT`, etc. |
| `quantity` | Number | Number of shares (0 for cash transactions) | `100` |
| `price` | Number | Price per share or cash amount | `130.50` |

## Optional Columns

| Column | Type | Description | Default |
|--------|------|-------------|---------|
| `fees` | Number | Transaction fees | `0.0` |

## Action Types

### Securities Transactions
- **BUY**: Purchase shares (requires `quantity > 0` and `price > 0`)
- **SELL**: Sell shares (requires `quantity > 0` and `price > 0`)

### Cash Transactions  
- **DEPOSIT**: Add cash to portfolio (use ticker `CASH`, `price` = amount)
- **WITHDRAWAL**: Remove cash (use ticker `CASH`, `price` = amount)
- **DIVIDEND**: Dividend payment (can use `quantity * price` or just `price`)
- **FEE**: Portfolio fee (use ticker `CASH`, `price` = fee amount)
- **INTEREST**: Interest earned (use ticker `CASH`, `price` = interest amount)

## Example File

```csv
date,ticker,action,quantity,price,fees
2023-01-03,CASH,DEPOSIT,0,100000,0
2023-01-03,AAPL,BUY,100,130,0
2023-01-03,MSFT,BUY,80,240,0
2023-06-30,AAPL,DIVIDEND,0,120,0  
2024-01-31,MSFT,BUY,5,350,0
```

## Validation Rules

1. **No short selling**: Cannot sell more shares than owned
2. **Valid dates**: All dates must be parseable (YYYY-MM-DD recommended)
3. **Positive values**: Quantity and price must be positive (or 0 for quantity in cash transactions)
4. **Ticker format**: Tickers auto-converted to uppercase, trimmed
5. **Column names**: Case-insensitive, auto-normalized to lowercase

## File Size Limits

- **Soft limit**: 500 rows (triggers warning in logs)
- **Hard limit**: 2000 rows (upload rejected with error message)

For portfolios exceeding 2000 rows, consider splitting into multiple upload batches.

## Notes

- **Column alias**: `shares` is automatically renamed to `quantity`
- **Amount calculation**: If `amount` column missing, calculated as `quantity * price`
- **Cash transactions**: Use ticker `CASH` for deposits, withdrawals, fees, and interest

## Sample Files

See example files in the repository:
- `sample_trades_full_metrics.csv` - Small example (9 rows)
- `large_portfolio_trades_contract_v1_bmonthend.csv` - Large example (1587 rows)

## Error Handling

### Common Errors

| Error Code | Description | Solution |
|------------|-------------|----------|
| `LEDGER_VALIDATION_FAILED` | Missing required columns or invalid data | Check column names match required fields |
| `FILE_TOO_LARGE` | File exceeds 2000 rows | Split into smaller files |
| Market data failures | Ticker not found or Yahoo Finance unavailable | Check ticker symbols, may see warning in response |

### Graceful Degradation

- **Missing tickers**: System continues processing valid tickers, returns warning for failed ones
- **Partial data**: Metrics computed with available data, coverage status indicates completeness
- **Invalid rows**: Specific validation errors returned with row numbers

## Related Documentation

- [METRICS_DEFINITIONS.md](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/docs/METRICS_DEFINITIONS.md) - Performance metric specifications
- [COVERAGE_SEMANTICS.md](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/docs/COVERAGE_SEMANTICS.md) - Data availability rules
- [MARKET_DATA.md](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/docs/MARKET_DATA.md) - Market data contracts and sources
