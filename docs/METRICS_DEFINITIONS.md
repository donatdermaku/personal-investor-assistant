# Metrics Definitions

This document is the canonical, test-backed source for metric definitions, inputs, and invariants.
Formulas are descriptive only; the implementation remains unchanged.

## Time-Weighted Return (TWR)
- Formula: daily r_t = (V_t - CF_t) / V_{t-1} - 1; TWR = prod(1 + r_t) - 1
- Time basis: daily
- Inputs: portfolio values, external cashflows
- Domain: requires at least 2 valuation points
- Invariants:
  - TWR == 0 when portfolio value is constant and external cashflows are zero
  - Daily returns are finite after normalization

## Money-Weighted Return (MWR / IRR)
- Formula: IRR on dated cashflows with terminal value as final inflow
- Time basis: daily
- Inputs: cashflows, terminal value
- Domain: requires at least 1 cashflow and terminal value
- Invariants:
  - Returns None when solver cannot converge
  - Equals 0 when terminal value equals net deposits over 1 year

## Volatility
- Formula: std dev of daily returns, annualized by sqrt(252) where specified
- Time basis: daily
- Inputs: daily returns
- Domain: requires >=2 daily return points
- Invariants:
  - Volatility >= 0
  - Cash-only portfolios yield volatility == 0

## Max Drawdown
- Formula: min(drawdown_t), drawdown_t = V_t / peak_t - 1
- Time basis: daily
- Inputs: portfolio values
- Domain: requires non-empty valuation series
- Invariants:
  - Max drawdown <= 0
  - Drawdown is 0 when values never fall below prior peaks

## Sharpe Ratio
- Formula: mean(excess returns) / std(returns) * sqrt(252)
- Time basis: daily
- Inputs: daily returns, risk-free rate
- Domain: requires >=2 daily return points and non-zero volatility
- Invariants:
  - Undefined when volatility is 0 (return None)

## Sortino Ratio
- Formula: mean(excess returns) / std(negative returns) * sqrt(252)
- Time basis: daily
- Inputs: daily returns, risk-free rate
- Domain: requires downside returns and non-zero downside deviation
- Invariants:
  - Undefined when downside deviation is 0 (return None)

## Value at Risk (Daily)
- Formula: VaR = quantile(returns, 1 - confidence)
- Time basis: daily
- Inputs: daily returns
- Domain: requires non-empty returns series
- Invariants:
  - VaR is finite when returns are finite

## Conditional VaR (Daily)
- Formula: CVaR = mean(returns <= VaR)
- Time basis: daily
- Inputs: daily returns, VaR
- Domain: requires non-empty returns series
- Invariants:
  - CVaR <= VaR

## Attribution Totals
- Formula: sum(component contributions) ~= total return
- Time basis: daily
- Inputs: component contributions, total return
- Domain: requires non-empty contributions
- Invariants:
  - Sum(components) ~= total return (within tolerance)

## Factor Tilts
- Formula: tilt = mean(portfolio factor pct) - mean(universe factor pct)
- Time basis: daily
- Inputs: portfolio scores, universe scores
- Domain: requires coverage in both sets
- Invariants:
  - Tilts are finite when input scores are finite
