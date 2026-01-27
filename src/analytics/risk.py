from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskContributionOutput:
    contributions: pd.DataFrame
    summary: dict


def compute_risk_contributions(
    returns: pd.DataFrame,
    weights: pd.Series,
    cash_weight: float = 0.0,
    var_alpha: float = 0.05,
) -> RiskContributionOutput:
    if returns.empty or weights.empty:
        return RiskContributionOutput(contributions=pd.DataFrame(), summary={})

    returns = returns.copy()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.reindex(returns.columns).fillna(0.0)
    if weights.sum() == 0:
        return RiskContributionOutput(contributions=pd.DataFrame(), summary={})

    cov = returns.cov()
    portfolio_var = float(weights.T @ cov @ weights)
    portfolio_vol = float(np.sqrt(portfolio_var)) if portfolio_var > 0 else 0.0

    if portfolio_vol == 0:
        return RiskContributionOutput(contributions=pd.DataFrame(), summary={})

    marginal = cov @ weights
    vol_contrib = weights * marginal / portfolio_vol
    vol_pct = vol_contrib / portfolio_vol if portfolio_vol != 0 else vol_contrib * 0.0

    portfolio_returns = returns @ weights
    var_value = float(np.quantile(portfolio_returns, var_alpha))
    if portfolio_returns.empty:
        var_value = 0.0
        var_contrib = pd.Series(0.0, index=weights.index)
    else:
        closest = (portfolio_returns - var_value).abs().idxmin()
        tail_returns = returns.loc[closest]
        base = float((weights * tail_returns).sum())
        scale = (var_value / base) if base != 0 else 0.0
        var_contrib = weights * tail_returns * scale

    var_pct = var_contrib / var_value if var_value != 0 else var_contrib * 0.0

    df = pd.DataFrame(
        {
            "ticker": weights.index,
            "volatility_contribution": vol_contrib.values,
            "volatility_pct": vol_pct.values,
            "var_contribution": var_contrib.values,
            "var_pct": var_pct.values,
        }
    ).sort_values("volatility_contribution", ascending=False)

    if cash_weight > 0:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "ticker": "CASH",
                            "volatility_contribution": 0.0,
                            "volatility_pct": 0.0,
                            "var_contribution": 0.0,
                            "var_pct": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    summary = {
        "portfolio_volatility": portfolio_vol,
        "portfolio_var": var_value,
        "var_alpha": var_alpha,
    }

    return RiskContributionOutput(contributions=df, summary=summary)


def compute_risk_contributions_from_cov(
    cov: np.ndarray | None,
    weights: pd.Series,
    tail_returns: np.ndarray | None,
    *,
    cash_weight: float = 0.0,
    var_alpha: float = 0.05,
    var_value: float | None = None,
) -> RiskContributionOutput:
    if cov is None or weights.empty:
        return RiskContributionOutput(contributions=pd.DataFrame(), summary={})

    weights = weights.fillna(0.0)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.size == 0:
        return RiskContributionOutput(contributions=pd.DataFrame(), summary={})

    w = weights.to_numpy(dtype=np.float64)
    portfolio_var = float(w.T @ cov @ w)
    portfolio_vol = float(np.sqrt(portfolio_var)) if portfolio_var > 0 else 0.0

    if portfolio_vol == 0:
        return RiskContributionOutput(contributions=pd.DataFrame(), summary={})

    marginal = cov @ w
    vol_contrib = w * marginal / portfolio_vol
    vol_pct = vol_contrib / portfolio_vol if portfolio_vol != 0 else vol_contrib * 0.0

    if var_value is None:
        var_value = 0.0
    if tail_returns is None or len(tail_returns) != len(w):
        var_contrib = np.zeros_like(w)
    else:
        base = float((w * tail_returns).sum())
        scale = (var_value / base) if base != 0 else 0.0
        var_contrib = w * tail_returns * scale

    var_pct = var_contrib / var_value if var_value != 0 else var_contrib * 0.0

    df = pd.DataFrame(
        {
            "ticker": weights.index,
            "volatility_contribution": vol_contrib,
            "volatility_pct": vol_pct,
            "var_contribution": var_contrib,
            "var_pct": var_pct,
        }
    ).sort_values("volatility_contribution", ascending=False)

    if cash_weight > 0:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "ticker": "CASH",
                            "volatility_contribution": 0.0,
                            "volatility_pct": 0.0,
                            "var_contribution": 0.0,
                            "var_pct": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    summary = {
        "portfolio_volatility": portfolio_vol,
        "portfolio_var": var_value,
        "var_alpha": var_alpha,
    }

    return RiskContributionOutput(contributions=df, summary=summary)
