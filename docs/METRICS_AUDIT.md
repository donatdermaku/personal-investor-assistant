# Metrics Audit

This table inventories metrics currently displayed in the Streamlit UI. It is a
source-of-truth index that links each metric to a `definition_key` in
`docs/DEFINITIONS.md` and `src/definitions.py`.

Columns:
- Metric: UI label or logical metric name.
- Definition Key: stable identifier for definitions.
- Where Shown: page/component.
- Source: function/module that computes or loads it.
- Inputs: data dependencies.
- Frequency: refresh cadence.
- Missingness Behavior: what the UI shows when data is missing.

| Metric | Definition Key | Where Shown | Source | Inputs | Frequency | Missingness Behavior |
| --- | --- | --- | --- | --- | --- | --- |
| Last pipeline run | last_pipeline_run | Header KPI | `src.streamlit_ui.build_status` | `prices_daily` (parquet) | Daily (pipeline) | `--` when prices empty |
| Price coverage | price_coverage | Header KPI | `src.streamlit_ui.build_status` | `prices_daily`, watchlist | Daily | `--` when prices empty |
| Fundamentals coverage | fundamentals_coverage | Header KPI | `src.streamlit_ui.build_status` | `scores_daily.has_fundamentals`, watchlist | Daily | `--` when scores empty |
| Portfolio source | portfolio_source | Header KPI | `src.portfolio.load_portfolio` | ledger/snapshot/demo | On load | `--` when no portfolio |
| Portfolio equity curve | portfolio_equity_curve | Dashboard/Performance | `src.portfolio.compute_portfolio_from_ledger/snapshot` | ledger/snapshot + `prices_daily` | On load | "No price data available." |
| Benchmark equity curve | benchmark_equity_curve | Dashboard/Performance | `src.streamlit_data.load_benchmark_prices` + scale | benchmark ticker + `prices_daily` or yfinance | On load | Hidden if benchmark series missing |
| Allocation weights | allocation_weights | Dashboard | `watchlist.yml` | `watchlist.yml` | On load | "No weights configured." |
| Drawdown series | drawdown_series | Dashboard | local calc in `pages/1_📊_Dashboard.py` | portfolio equity curve | On load | "No drawdown data available." |
| Max drawdown | max_drawdown | Dashboard | `src.intelligence.drawdown_intelligence` | portfolio equity curve | On load | Not shown if values empty |
| Current drawdown | current_drawdown | Dashboard | `src.intelligence.drawdown_intelligence` | portfolio equity curve | On load | Not shown if values empty |
| Factor tilts | factor_tilts | Dashboard | `src.intelligence.factor_tilts` | `scores_daily` percentiles | Daily | "No tilt data available." |
| TWR — Strategy Return | twr | Performance | `src.portfolio.compute_portfolio_from_ledger/snapshot` | ledger/snapshot + prices | On load | `--` if unavailable |
| MWR — Your Personal Return | mwr | Performance (Pro) | `src.portfolio._xirr` | ledger cashflows + terminal value | On load | `--` if unavailable |
| Attribution (30d) | attribution_30d | Performance | local calc in `pages/5_📈_Performance.py` | 30d mean returns + weights | On load | "Insufficient price history." |
| Monthly return heatmap | monthly_return | Performance | local calc in `pages/5_📈_Performance.py` | portfolio daily returns | On load | "Insufficient data for heatmap." |
| Rolling volatility | rolling_volatility | Performance | local calc in `pages/5_📈_Performance.py` | portfolio daily returns | On load | Empty chart if no returns |
| Rolling Sharpe | rolling_sharpe | Performance | local calc in `pages/5_📈_Performance.py` | portfolio daily returns | On load | Empty chart if no returns |
| Component risk | component_risk | Performance/Risk | `src.intelligence.component_risk` | returns + weights | On load | "No component risk data available." |
| Correlation matrix | correlation_matrix | Risk | local calc in `pages/4_⚠️_Risk_Management.py` | returns | On load | "Insufficient price history." |
| VaR (daily) | var_daily | Risk | local calc in `pages/4_⚠️_Risk_Management.py` | portfolio returns | On load | `nan` if returns empty |
| CVaR (daily) | cvar_daily | Risk | local calc in `pages/4_⚠️_Risk_Management.py` | portfolio returns | On load | `nan` if returns empty |
| Stress test impact | stress_test_impact | Risk | local calc in `pages/4_⚠️_Risk_Management.py` | latest prices | On load | "Insufficient price history." |
| Position sizing (shares) | position_sizing_shares | Risk | local calc in `pages/4_⚠️_Risk_Management.py` | price + risk inputs | On load | "Price not available for sizing." |
| Price | price_spot | Watchlist/Research | `scores_daily.Price` | `scores_daily` | Daily | `--` if ticker missing |
| Composite percentile | composite_pct | Watchlist/Research/Peers | `scores_daily.composite_pct` | `scores_daily` | Daily | Empty when scores missing |
| Value percentile | value_pct | Watchlist/Research/Peers | `scores_daily.value_pct` | `scores_daily` | Daily | Empty when scores missing |
| Quality percentile | quality_pct | Watchlist/Research/Peers | `scores_daily.quality_pct` | `scores_daily` | Daily | Empty when scores missing |
| Momentum percentile | momentum_pct | Watchlist/Research/Peers | `scores_daily.momentum_pct` | `scores_daily` | Daily | Empty when scores missing |
| Piotroski F | piotroski_f | Watchlist | `scores_daily.PiotroskiF` | `scores_daily` | Daily | Empty when scores missing |
| Volatility 30d | volatility_30d | Watchlist | `scores_daily.Volatility30d` | `scores_daily` | Daily | Empty when scores missing |
| Sharpe 1y | sharpe_1y | Watchlist | `scores_daily.Sharpe1y` | `scores_daily` | Daily | Empty when scores missing |
| Industry | industry | Watchlist/Peers | `scores_daily.industry` | `scores_daily` | Daily | Empty when scores missing |
| SMA 20 | sma_20 | Stock Research | local calc in `pages/3_🔍_Stock_Research.py` | `prices_daily.adj_close` | Daily | Not shown if price history missing |
| SMA 50 | sma_50 | Stock Research | local calc in `pages/3_🔍_Stock_Research.py` | `prices_daily.adj_close` | Daily | Not shown if price history missing |
| RSI 14 | rsi_14 | Stock Research | local calc in `pages/3_🔍_Stock_Research.py` | `prices_daily.adj_close` | Daily | Not shown if price history missing |
| Revenue | fundamentals_revenue | Stock Research | `fundamentals_quarterly.Revenue` | `fundamentals_quarterly` | Quarterly | "No fundamentals history." |
| Net income | fundamentals_net_income | Stock Research | `fundamentals_quarterly.NetIncome` | `fundamentals_quarterly` | Quarterly | "No fundamentals history." |
| Operating CF | fundamentals_operating_cf | Stock Research | `fundamentals_quarterly.OperatingCF` | `fundamentals_quarterly` | Quarterly | "No fundamentals history." |
| CapEx | fundamentals_capex | Stock Research | `fundamentals_quarterly.CapitalExpenditures` | `fundamentals_quarterly` | Quarterly | "No fundamentals history." |
| Total assets | fundamentals_total_assets | Stock Research | `fundamentals_quarterly.TotalAssets` | `fundamentals_quarterly` | Quarterly | "No fundamentals history." |
| Total liabilities | fundamentals_total_liabilities | Stock Research | `fundamentals_quarterly.TotalLiabilities` | `fundamentals_quarterly` | Quarterly | "No fundamentals history." |
| Latest daily return | latest_daily_return | Guidance | `src.guidance.portfolio_change_summary` | portfolio daily returns | On load | "Portfolio return data unavailable." |
| 7-day return | return_7d | Guidance | `src.guidance.portfolio_change_summary` | portfolio daily returns | On load | `--` if <7 points |
| Composite % change 1d | composite_pct_change_1d | Guidance | `src.guidance.explain_ticker_change` | `scores_daily` | Daily | Skipped if NaN |
| Composite % change 7d | composite_pct_change_7d | Guidance | `src.guidance.explain_ticker_change` | `scores_daily` | Daily | Skipped if NaN |
| Drawdown 1y | drawdown_1y | Guidance | `scores_daily.Drawdown1y` | `scores_daily` | Daily | Skipped if NaN |
| Risk warnings | risk_warnings | Guidance | `src.guidance.risk_warnings` | returns + scores | On load | Empty list if no triggers |
