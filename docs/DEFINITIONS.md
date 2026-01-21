# Definitions Registry

This document defines formulas and assumptions for metrics used in the Streamlit UI.
Keys here map to `src/definitions.py` and `docs/METRICS_AUDIT.md`.

## twr
Time-Weighted Return (TWR) is computed from daily portfolio values and external
cashflows:

- Daily return: `r_t = (V_t - CF_t) / V_{t-1} - 1`
  - `V_t` is total portfolio value on day `t` (holdings + cash).
  - `CF_t` is net external cashflow on day `t` (DEPOSIT/WITHDRAWAL only).
- TWR: `(1 + r_t).prod() - 1` over the available series.

Assumptions:
- Uses adjusted close from `prices_daily`.
- Missing `V_{t-1}` yields `r_t = 0` after NaN replacement.
- Ledger mode includes cash balance and trades; snapshot mode uses holdings only.

## mwr
Money-Weighted Return (MWR) uses an XIRR-style IRR on dated cashflows:

- Cashflows include DEPOSIT, WITHDRAWAL, DIVIDEND, FEE, INTEREST.
- Terminal value is the most recent portfolio value appended as a final inflow.
- Numerical solver uses Newton iterations with a finite-difference derivative.

If the solver does not converge, MWR is `None` and the UI displays `--`.

## benchmark_alignment
Benchmark series are not strictly calendar-aligned to portfolio values. The
benchmark is scaled to the portfolio at the first available benchmark date:

`benchmark_scaled = (benchmark_adj_close / benchmark_adj_close[0]) * V_0`

The chart overlays the scaled series with the portfolio equity curve.

## max_drawdown / current_drawdown
Drawdown is computed on the portfolio equity curve:

- `peak_t = max(V_0..V_t)`
- `drawdown_t = V_t / peak_t - 1`
- `max_drawdown = min(drawdown_t)`
- `current_drawdown = drawdown_t` at the latest date

## rolling_volatility
Rolling volatility (portfolio) is the rolling standard deviation of daily
portfolio returns over the selected window. The UI shows raw daily volatility
(no annualization applied in the UI chart).

## sharpe_rolling
Rolling Sharpe uses the same windowed mean and standard deviation:

`sharpe_t = (mean(r_t) / std(r_t)) * sqrt(252)`

If standard deviation is zero or NaN, Sharpe is NaN.

## var_daily / cvar_daily
Value at Risk (VaR) and Conditional VaR (CVaR) use the empirical distribution
of daily portfolio returns at confidence level `c`:

- `VaR = quantile(returns, 1 - c)`
- `CVaR = mean(returns[returns <= VaR])`

If returns are missing, values may be NaN.

## factor_tilts
Factor tilt compares mean portfolio percentiles to universe means:

`tilt = mean(portfolio_factor_pct) - mean(universe_factor_pct)`

Portfolio means are computed over watchlist tickers. Universe means are computed
over all `scores_daily` rows.

## component_risk
Component risk contribution uses the covariance matrix:

- `cov = returns.cov()`
- `portfolio_var = w.T @ cov @ w`
- `contrib = (w * (cov @ w)) / portfolio_var`

Weights are normalized if possible; empty weights yield no output.

## attribution_30d
Attribution (30d) is a simple contribution proxy:

`contrib_i = mean(returns_i over last 30 days) * weight_i`

Weights come from `watchlist.yml` or equal-weight if missing.

## price_spot
Price is the latest adjusted close in `scores_daily.Price` (from the pipeline).

## composite_pct / value_pct / quality_pct / momentum_pct
Percentiles are computed in the pipeline as percentile ranks over the universe:

`pct = rank(score, pct=True) * 100`

These are stored in `scores_daily` as `*_pct`.

## piotroski_f
Piotroski F-score is computed in the pipeline using the nine-signal checklist.

## volatility_30d / sharpe_1y / drawdown_1y
These are computed in the pipeline from adjusted-close returns:

- `volatility_30d`: std of last ~30 daily returns, annualized by `sqrt(252)`
- `sharpe_1y`: mean/vol over last ~252 daily returns, annualized by `sqrt(252)`
- `drawdown_1y`: last price vs max over trailing ~252 sessions

## sma_20 / sma_50 / rsi_14
Technical indicators computed in Stock Research:

- SMA: rolling mean of adjusted close over 20/50 days.
- RSI 14: standard 14-day RSI on adjusted close.

## stress_test_impact
Stress test multiplies the latest price by `(1 + shock)` and reports the
percentage change per ticker.

## position_sizing_shares
Position sizing uses a fixed-risk formula:

`risk_amount = portfolio_value * (risk_pct / 100)`
`shares = risk_amount / (price * (stop_loss_pct / 100))`

## latest_daily_return / return_7d
Guidance summary metrics:

- Latest daily return: last portfolio daily return.
- 7-day return: compounded return over last 7 daily returns.
