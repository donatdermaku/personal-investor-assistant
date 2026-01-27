from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.streaming import build_canonical_calendar, iter_portfolio_state


@dataclass
class AttributionOutput:
    summary: dict
    timeseries: pd.DataFrame
    per_asset: pd.DataFrame


def compute_attribution(
    prices: pd.DataFrame,
    holdings_daily: pd.DataFrame,
    daily_values: pd.DataFrame,
    portfolio_returns: pd.Series,
    *,
    calendar: pd.DatetimeIndex | None = None,
) -> AttributionOutput:
    if prices.empty or holdings_daily.empty or daily_values.empty:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    prices = prices.dropna(subset=["date", "ticker", "adj_close"])
    holdings = holdings_daily.dropna(subset=["date", "ticker", "quantity"])
    if prices.empty or holdings.empty:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    tickers = sorted(prices["ticker"].dropna().unique().tolist())
    if not tickers:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())
    if calendar is None:
        calendar = build_canonical_calendar(prices, total_values=daily_values)
    if calendar.empty:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    portfolio_returns = portfolio_returns.copy()
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index, errors="coerce").normalize()
    portfolio_returns = portfolio_returns.reindex(calendar).fillna(0.0)

    n_assets = len(tickers)
    bench_weight = 1.0 / n_assets
    allocation = np.zeros(len(calendar), dtype=np.float64)
    selection = np.zeros(len(calendar), dtype=np.float64)
    interaction = np.zeros(len(calendar), dtype=np.float64)

    prev_weights = np.zeros(n_assets, dtype=np.float32)
    for idx, (_, returns, weights) in enumerate(
        iter_portfolio_state(prices, holdings, daily_values, tickers, calendar)
    ):
        bench_return = float(np.mean(returns)) if n_assets else 0.0
        allocation_by_asset = prev_weights * bench_return
        selection_by_asset = (returns - bench_return) * bench_weight
        interaction_by_asset = (prev_weights - bench_weight) * (returns - bench_return)

        allocation[idx] = float(allocation_by_asset.sum())
        selection[idx] = float(selection_by_asset.sum())
        interaction[idx] = float(interaction_by_asset.sum())
        prev_weights = weights

    raw_total = allocation + selection + interaction
    scale = np.ones(len(calendar), dtype=np.float64)
    mask = raw_total != 0
    scale[mask] = portfolio_returns.to_numpy(dtype=np.float64)[mask] / raw_total[mask]

    allocation_scaled = allocation * scale
    selection_scaled = selection * scale
    interaction_scaled = interaction * scale

    timeseries = pd.DataFrame(
        {
            "date": calendar.strftime("%Y-%m-%d"),
            "allocation": allocation_scaled,
            "selection": selection_scaled,
            "interaction": interaction_scaled,
            "total_return": portfolio_returns.values,
        }
    )

    summary = _summarize_attribution(
        pd.Series(allocation_scaled, index=calendar),
        pd.Series(selection_scaled, index=calendar),
        pd.Series(interaction_scaled, index=calendar),
        portfolio_returns,
    )

    # Per-asset summaries (streamed)
    log_terms = np.log1p(portfolio_returns.to_numpy(dtype=np.float64))
    log_terms = np.where(portfolio_returns.to_numpy() == -1.0, np.nan, log_terms)
    sum_log = float(np.nansum(log_terms))
    total_return = float(np.expm1(sum_log)) if sum_log != 0 else 0.0
    carino_weights = np.zeros(len(calendar), dtype=np.float64)
    pr_vals = portfolio_returns.to_numpy(dtype=np.float64)
    non_zero = pr_vals != 0
    carino_weights[non_zero] = log_terms[non_zero] / pr_vals[non_zero]
    factor = np.log1p(total_return) / sum_log if sum_log != 0 else 1.0

    alloc_sum = np.zeros(n_assets, dtype=np.float64)
    sel_sum = np.zeros(n_assets, dtype=np.float64)
    int_sum = np.zeros(n_assets, dtype=np.float64)

    prev_weights = np.zeros(n_assets, dtype=np.float32)
    for idx, (_, returns, weights) in enumerate(
        iter_portfolio_state(prices, holdings, daily_values, tickers, calendar)
    ):
        bench_return = float(np.mean(returns)) if n_assets else 0.0
        allocation_by_asset = prev_weights * bench_return
        selection_by_asset = (returns - bench_return) * bench_weight
        interaction_by_asset = (prev_weights - bench_weight) * (returns - bench_return)

        scale_t = scale[idx]
        weight_t = carino_weights[idx]
        if weight_t != 0 and scale_t != 0:
            alloc_sum += allocation_by_asset * scale_t * weight_t
            sel_sum += selection_by_asset * scale_t * weight_t
            int_sum += interaction_by_asset * scale_t * weight_t
        prev_weights = weights

    per_asset = pd.DataFrame(
        {
            "ticker": tickers,
            "allocation": alloc_sum * factor,
            "selection": sel_sum * factor,
            "interaction": int_sum * factor,
        }
    )
    if not per_asset.empty:
        per_asset["total"] = per_asset["allocation"] + per_asset["selection"] + per_asset["interaction"]
        per_asset = per_asset.sort_values("total", ascending=False)

    summary["per_asset"] = per_asset.to_dict(orient="records") if not per_asset.empty else []
    summary["method"] = "carino"
    summary["benchmark"] = "equal_weight"

    return AttributionOutput(summary=summary, timeseries=timeseries, per_asset=per_asset)


def _carino_link(component: pd.Series, total_returns: pd.Series) -> float:
    total_returns = total_returns.fillna(0.0)
    component = component.reindex(total_returns.index).fillna(0.0)

    log_terms = np.log1p(total_returns.replace({-1.0: np.nan}))
    sum_log = float(log_terms.sum())
    total_return = float(np.expm1(sum_log)) if sum_log != 0 else 0.0

    if sum_log == 0:
        return float(component.sum())

    weights = pd.Series(0.0, index=total_returns.index)
    non_zero = total_returns != 0
    weights[non_zero] = log_terms[non_zero] / total_returns[non_zero]

    scaled = float((component * weights).sum())
    factor = np.log1p(total_return) / sum_log if sum_log != 0 else 1.0
    return float(scaled * factor)


def _summarize_attribution(
    allocation: pd.Series,
    selection: pd.Series,
    interaction: pd.Series,
    total_returns: pd.Series,
) -> dict:
    allocation_total = _carino_link(allocation, total_returns)
    selection_total = _carino_link(selection, total_returns)
    interaction_total = _carino_link(interaction, total_returns)
    total_return = float(np.expm1(np.log1p(total_returns.fillna(0.0)).sum()))
    combined = allocation_total + selection_total + interaction_total
    if combined != 0:
        scale = total_return / combined
        allocation_total *= scale
        selection_total *= scale
        interaction_total *= scale
    return {
        "total_return": total_return,
        "allocation": allocation_total,
        "selection": selection_total,
        "interaction": interaction_total,
    }


def _summarize_by_asset(
    allocation: pd.DataFrame,
    selection: pd.DataFrame,
    interaction: pd.DataFrame,
    total_returns: pd.Series,
) -> pd.DataFrame:
    rows = []
    for ticker in allocation.columns:
        alloc_total = _carino_link(allocation[ticker], total_returns)
        sel_total = _carino_link(selection[ticker], total_returns)
        int_total = _carino_link(interaction[ticker], total_returns)
        rows.append(
            {
                "ticker": ticker,
                "allocation": alloc_total,
                "selection": sel_total,
                "interaction": int_total,
                "total": alloc_total + sel_total + int_total,
            }
        )
    return pd.DataFrame(rows).sort_values("total", ascending=False)
