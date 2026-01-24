from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
) -> AttributionOutput:
    if prices.empty or holdings_daily.empty or daily_values.empty:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker", "adj_close"])

    holdings = holdings_daily.copy()
    holdings["date"] = pd.to_datetime(holdings["date"], errors="coerce")
    holdings = holdings.dropna(subset=["date", "ticker", "quantity"])

    if prices.empty or holdings.empty:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    price_wide = (
        prices.pivot_table(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .ffill()
    )
    returns = price_wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    holdings_wide = holdings.pivot_table(index="date", columns="ticker", values="quantity").sort_index()
    holdings_wide = holdings_wide.reindex(index=price_wide.index, columns=price_wide.columns).fillna(0.0)

    holdings_value = holdings_wide.mul(price_wide, axis=1)
    total_value = daily_values["value"].reindex(price_wide.index).fillna(0.0)
    weights = holdings_value.div(total_value.replace({0: np.nan}), axis=0).fillna(0.0)

    aligned_returns = returns.reindex(weights.index).fillna(0.0)
    weights = weights.shift(1).reindex(aligned_returns.index).fillna(0.0)

    if weights.empty:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    tickers = list(weights.columns)
    if not tickers:
        return AttributionOutput(summary={}, timeseries=pd.DataFrame(), per_asset=pd.DataFrame())

    bench_weights = pd.Series(1.0 / len(tickers), index=tickers)
    bench_return = aligned_returns.mul(bench_weights, axis=1).sum(axis=1)

    allocation_by_asset = weights.mul(bench_return, axis=0)
    selection_by_asset = (aligned_returns.sub(bench_return, axis=0)).mul(bench_weights, axis=1)
    interaction_by_asset = (weights.sub(bench_weights, axis=1)).mul(aligned_returns.sub(bench_return, axis=0), axis=1)

    allocation = allocation_by_asset.sum(axis=1)
    selection = selection_by_asset.sum(axis=1)
    interaction = interaction_by_asset.sum(axis=1)
    raw_total = allocation + selection + interaction

    portfolio_returns = portfolio_returns.reindex(raw_total.index).fillna(0.0)
    scale = pd.Series(0.0, index=raw_total.index)
    mask = raw_total != 0
    scale[mask] = portfolio_returns[mask] / raw_total[mask]

    allocation = allocation.mul(scale, axis=0)
    selection = selection.mul(scale, axis=0)
    interaction = interaction.mul(scale, axis=0)

    allocation_by_asset = allocation_by_asset.mul(scale, axis=0)
    selection_by_asset = selection_by_asset.mul(scale, axis=0)
    interaction_by_asset = interaction_by_asset.mul(scale, axis=0)

    timeseries = pd.DataFrame(
        {
            "date": raw_total.index.strftime("%Y-%m-%d"),
            "allocation": allocation.values,
            "selection": selection.values,
            "interaction": interaction.values,
            "total_return": portfolio_returns.values,
        }
    )

    summary = _summarize_attribution(
        allocation,
        selection,
        interaction,
        portfolio_returns,
    )

    per_asset = _summarize_by_asset(
        allocation_by_asset,
        selection_by_asset,
        interaction_by_asset,
        portfolio_returns,
    )

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
