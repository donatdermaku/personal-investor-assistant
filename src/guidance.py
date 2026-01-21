from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GuidanceSummary:
    what_changed: list[str]
    why: list[str]
    next_steps: list[str]
    risk_warnings: list[str]


def _safe_pct(value: float | None) -> str:
    if value is None or value != value:
        return "--"
    return f"{value:.2%}"


def portfolio_change_summary(portfolio_returns: pd.Series) -> list[str]:
    if portfolio_returns.empty:
        return ["Portfolio return data unavailable."]
    last = portfolio_returns.iloc[-1]
    last_7 = (1 + portfolio_returns.tail(7)).prod() - 1 if len(portfolio_returns) >= 7 else np.nan
    return [
        f"Latest daily return: {_safe_pct(last)}",
        f"7-day return: {_safe_pct(last_7)}",
    ]


def risk_warnings(
    portfolio_returns: pd.Series,
    scores: pd.DataFrame,
    watch_tickers: list[str],
) -> list[str]:
    warnings: list[str] = []
    if not portfolio_returns.empty:
        vol = portfolio_returns.tail(30).std()
        if vol == vol and vol > 0.03:
            warnings.append("Risk increased due to elevated 30d volatility.")
        drawdown = _latest_drawdown(portfolio_returns)
        if drawdown is not None and drawdown < -0.15:
            warnings.append("Drawdown remains elevated; worth reviewing exposure.")
    if not scores.empty and watch_tickers:
        subset = scores[scores["ticker"].isin(watch_tickers)]
        if not subset.empty:
            low_quality = subset[subset["quality_pct"] < 30]
            if not low_quality.empty:
                warnings.append("Several holdings rank low on quality versus the universe.")
            weak_momentum = subset[subset["momentum_pct"] < 30]
            if not weak_momentum.empty:
                warnings.append("Momentum has weakened for part of the watchlist.")
    return warnings


def explain_ticker_change(ticker: str, scores: pd.DataFrame) -> GuidanceSummary:
    if scores.empty or ticker not in scores["ticker"].values:
        return GuidanceSummary(["No data for selected ticker."], [], [], [])
    row = scores.set_index("ticker").loc[ticker]
    change_1d = row.get("composite_pct_change_1d")
    change_7d = row.get("composite_pct_change_7d")
    what = []
    if change_1d == change_1d:
        what.append(f"Composite percentile moved {change_1d:+.1f} pts over 1d.")
    if change_7d == change_7d:
        what.append(f"Composite percentile moved {change_7d:+.1f} pts over 7d.")

    why = []
    value_pct = row.get("value_pct")
    quality_pct = row.get("quality_pct")
    momentum_pct = row.get("momentum_pct")
    if value_pct == value_pct:
        why.append(f"Value percentile: {value_pct:.1f}.")
    if quality_pct == quality_pct:
        why.append(f"Quality percentile: {quality_pct:.1f}.")
    if momentum_pct == momentum_pct:
        why.append(f"Momentum percentile: {momentum_pct:.1f}.")

    next_steps = [
        "Worth reviewing recent price action and fundamentals for confirmation.",
    ]

    risks = []
    drawdown = row.get("Drawdown1y")
    if drawdown == drawdown and drawdown < -0.2:
        risks.append("Drawdown exceeds 20%; risk profile elevated.")

    return GuidanceSummary(what, why, next_steps, risks)


def explain_portfolio(scores: pd.DataFrame, portfolio_returns: pd.Series, watch_tickers: list[str]) -> GuidanceSummary:
    what = portfolio_change_summary(portfolio_returns)
    why = []
    if not scores.empty and watch_tickers:
        subset = scores[scores["ticker"].isin(watch_tickers)]
        if not subset.empty:
            top = subset.sort_values("composite_pct", ascending=False).head(3)
            why.append("Top composite contributors: " + ", ".join(top["ticker"].tolist()))
    next_steps = ["Worth reviewing outliers in volatility and drawdown."]
    risks = risk_warnings(portfolio_returns, scores, watch_tickers)
    return GuidanceSummary(what, why, next_steps, risks)


def _latest_drawdown(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    index = (1 + returns).cumprod()
    peak = index.cummax()
    dd = index / peak - 1.0
    return float(dd.iloc[-1]) if not dd.empty else None
