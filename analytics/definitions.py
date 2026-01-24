METRICS_REGISTRY = {
    "twr": {
        "title": "Time-Weighted Return (TWR)",
        "formula": "Daily r_t = (V_t - CF_t) / V_{t-1} - 1; TWR = prod(1 + r_t) - 1.",
        "time_basis": "daily",
        "inputs": ["portfolio_values", "external_cashflows"],
        "domain": "Requires at least 2 valuation points.",
        "invariants": [
            "TWR == 0 when portfolio value is constant and external cashflows are zero.",
            "Daily returns are finite (no NaN/inf) after normalization.",
        ],
    },
    "mwr": {
        "title": "Money-Weighted Return (MWR / IRR)",
        "formula": "IRR on dated cashflows with terminal portfolio value as final inflow.",
        "time_basis": "daily",
        "inputs": ["cashflows", "terminal_value"],
        "domain": "Requires at least 1 cashflow and terminal value.",
        "invariants": [
            "Returns None when solver cannot converge.",
            "MWR equals 0 when terminal value equals net deposits over 1 year.",
        ],
    },
    "volatility": {
        "title": "Volatility",
        "formula": "Std dev of daily returns; annualized by sqrt(252) where specified.",
        "time_basis": "daily",
        "inputs": ["daily_returns"],
        "domain": "Requires >=2 daily return points.",
        "invariants": [
            "Volatility >= 0.",
            "Cash-only portfolios yield volatility == 0.",
        ],
    },
    "max_drawdown": {
        "title": "Max Drawdown",
        "formula": "min(drawdown_t), drawdown_t = V_t / peak_t - 1.",
        "time_basis": "daily",
        "inputs": ["portfolio_values"],
        "domain": "Requires non-empty valuation series.",
        "invariants": [
            "Max drawdown <= 0.",
            "Drawdown is 0 when values never fall below prior peaks.",
        ],
    },
    "sharpe": {
        "title": "Sharpe Ratio",
        "formula": "mean(excess returns) / std(returns) * sqrt(252).",
        "time_basis": "daily",
        "inputs": ["daily_returns", "risk_free_rate"],
        "domain": "Requires >=2 daily return points and non-zero volatility.",
        "invariants": [
            "Undefined when volatility is 0 (return None).",
        ],
    },
    "sortino": {
        "title": "Sortino Ratio",
        "formula": "mean(excess returns) / std(negative returns) * sqrt(252).",
        "time_basis": "daily",
        "inputs": ["daily_returns", "risk_free_rate"],
        "domain": "Requires downside returns and non-zero downside deviation.",
        "invariants": [
            "Undefined when downside deviation is 0 (return None).",
        ],
    },
    "var_daily": {
        "title": "Value at Risk (Daily)",
        "formula": "VaR = quantile(returns, 1 - confidence).",
        "time_basis": "daily",
        "inputs": ["daily_returns"],
        "domain": "Requires non-empty returns series.",
        "invariants": [
            "VaR is finite when returns are finite.",
        ],
    },
    "cvar_daily": {
        "title": "Conditional VaR (Daily)",
        "formula": "CVaR = mean(returns <= VaR).",
        "time_basis": "daily",
        "inputs": ["daily_returns", "var_daily"],
        "domain": "Requires non-empty returns series.",
        "invariants": [
            "CVaR <= VaR (more conservative tail loss).",
        ],
    },
    "attribution_totals": {
        "title": "Attribution Totals",
        "formula": "Sum of component contributions equals total return (within tolerance).",
        "time_basis": "daily",
        "inputs": ["component_contributions", "total_return"],
        "domain": "Requires non-empty contributions.",
        "invariants": [
            "Sum(components) ~= total return.",
        ],
    },
    "allocation_effect": {
        "title": "Allocation Effect",
        "formula": "Allocation = sum(weights * benchmark_return).",
        "time_basis": "daily",
        "inputs": ["weights", "benchmark_return"],
        "domain": "Requires asset weights and benchmark return.",
        "invariants": [
            "Allocation contribution is finite when inputs are finite.",
        ],
    },
    "selection_effect": {
        "title": "Selection Effect",
        "formula": "Selection = sum(benchmark_weights * (asset_return - benchmark_return)).",
        "time_basis": "daily",
        "inputs": ["benchmark_weights", "asset_returns", "benchmark_return"],
        "domain": "Requires asset returns and benchmark return.",
        "invariants": [
            "Selection contribution is finite when inputs are finite.",
        ],
    },
    "interaction_effect": {
        "title": "Interaction Effect",
        "formula": "Interaction = sum((weights - benchmark_weights) * (asset_return - benchmark_return)).",
        "time_basis": "daily",
        "inputs": ["weights", "benchmark_weights", "asset_returns", "benchmark_return"],
        "domain": "Requires asset returns and weights.",
        "invariants": [
            "Interaction contribution is finite when inputs are finite.",
        ],
    },
    "tracking_error": {
        "title": "Tracking Error",
        "formula": "Std dev of active returns, annualized by sqrt(252).",
        "time_basis": "daily",
        "inputs": ["portfolio_returns", "benchmark_returns"],
        "domain": "Requires overlapping return series.",
        "invariants": [
            "Tracking error >= 0.",
        ],
    },
    "benchmark_volatility": {
        "title": "Benchmark Volatility",
        "formula": "Std dev of benchmark daily returns, annualized.",
        "time_basis": "daily",
        "inputs": ["benchmark_returns"],
        "domain": "Requires benchmark returns.",
        "invariants": [
            "Benchmark volatility >= 0.",
        ],
    },
    "benchmark_correlation": {
        "title": "Benchmark Correlation",
        "formula": "Correlation of portfolio and benchmark daily returns.",
        "time_basis": "daily",
        "inputs": ["portfolio_returns", "benchmark_returns"],
        "domain": "Requires overlapping return series.",
        "invariants": [
            "Correlation in [-1, 1] when defined.",
        ],
    },
    "rolling_drawdown": {
        "title": "Rolling Drawdown",
        "formula": "Rolling minimum of drawdown over the window.",
        "time_basis": "daily",
        "inputs": ["drawdown_series"],
        "domain": "Requires drawdown series and window.",
        "invariants": [
            "Rolling drawdown <= 0.",
        ],
    },
    "factor_tilts": {
        "title": "Factor Tilts",
        "formula": "Tilt = mean(portfolio factor pct) - mean(universe factor pct).",
        "time_basis": "daily",
        "inputs": ["portfolio_scores", "universe_scores"],
        "domain": "Requires portfolio and universe factor coverage.",
        "invariants": [
            "Tilts are finite when input scores are finite.",
        ],
    },
}
