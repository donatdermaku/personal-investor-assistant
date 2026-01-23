from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.nexus_ui import render_layout, render_kpi_card
from src.streamlit_data import (
    get_fundamentals,
    get_prices,
    get_scores,
    load_benchmark_prices,
    load_portfolio_cached,
    load_watchlist,
    market_status,
    portfolio_cache_token,
)
from src.streamlit_ui import build_status, render_portfolio_errors
from src.guidance import explain_portfolio
from src.glossary import GLOSSARY
from src.intelligence import component_risk
from src.portfolio import align_benchmark, compute_monthly_returns

st.set_page_config(page_title="Performance - Nexus Analytics", page_icon="📈", layout="wide")

def sidebar_content():
    st.write("Performance Settings")
    st.slider("Rolling Window", 30, 252, 60, key="perf_rolling_window")

def context_content(twr, mwr, last_run):
    st.markdown("### Returns")
    render_kpi_card("TWR", f"{twr:.2%}" if twr is not None else "--", "Strategy Return", tooltip="Time-Weighted Return")
    render_kpi_card("MWR", f"{mwr:.2%}" if mwr is not None else "--", "Personal Return", tooltip="Money-Weighted Return")
    
    st.divider()
    st.caption(f"Last updated: {last_run}")
    
    # Analysis Summary
    st.markdown("**Intelligence**")
    st.info("Analysis summary will appear here in Phase 12.4.")

def main_content(portfolio, prices, bench_prices, benchmark, watch_tickers, weights):
    if portfolio.daily_values.empty:
        st.warning("No data available.")
        return

    # 1. Equity Curve
    st.write("### Equity Curve vs Benchmark")
    eq = portfolio.daily_values["value"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio", line=dict(color="#0F172A", width=2)))
    if not bench_prices.empty:
        bench_index = align_benchmark(bench_prices, eq)
        if not bench_index.empty:
            fig.add_trace(go.Scatter(x=bench_index.index, y=bench_index.values, name=benchmark, line=dict(color="#9CA3AF", width=1.5, dash="dot")))
    
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Attribution
    st.write("### Attribution (Top Contributors)")
    # Logic copied from original page
    prices_sub = prices[prices["ticker"].isin(watch_tickers)].copy()
    if not prices_sub.empty:
        prices_sub["date"] = pd.to_datetime(prices_sub["date"])
        wide = prices_sub.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
        wide = wide.ffill().dropna(how="all")
        returns = wide.pct_change().dropna()
        
        # Recalc weights if missing
        w_series = pd.Series(weights).reindex(returns.columns).fillna(0.0)
        if w_series.sum() == 0:
            w_series[:] = 1.0 / len(w_series)
        else:
            w_series = w_series / w_series.sum()
            
        recent = returns.tail(30)
        contrib = recent.mean() * w_series
        contrib = contrib.sort_values(ascending=False).head(10) # Top 10
        
        fig2 = go.Figure(data=[go.Bar(x=contrib.index, y=contrib.values, marker_color="#3B82F6")])
        fig2.update_layout(template="plotly_white", height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Monthly Heatmap
    st.write("### Monthly Returns")
    monthly = compute_monthly_returns(portfolio.daily_returns)
    if not monthly.empty:
        heat = monthly.to_frame("return")
        heat["year"] = heat.index.year
        heat["month"] = heat.index.month
        pivot = heat.pivot(index="year", columns="month", values="return").fillna(0)
        
        fig3 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(m) for m in pivot.columns],
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmin=-0.1, zmax=0.1
        ))
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)


def main():
    watchlist = load_watchlist()
    watch_tickers = watchlist.get("tickers", [])
    weights = watchlist.get("weights", {})
    _, market_state = market_status()
    
    # Load Data
    prices, price_meta = get_prices(market_state, watch_tickers)
    scores, scores_meta = get_scores(watch_tickers)
    _, fund_meta = get_fundamentals(watch_tickers)
    
    portfolio = load_portfolio_cached(
        prices,
        watch_tickers,
        portfolio_cache_token(),
        source_override=st.session_state.get("portfolio_source"),
        uploads_active=st.session_state.get("uploads_active", False)
    )
    
    benchmark = "SPY"
    bench_prices = load_benchmark_prices(benchmark)
    
    status = build_status(price_meta, fund_meta, portfolio)
    
    render_layout(
        "Performance",
        sidebar_content,
        lambda: main_content(portfolio, prices, bench_prices, benchmark, watch_tickers, weights),
        lambda: context_content(portfolio.twr, portfolio.mwr, status["last_run"])
    )

if __name__ == "__main__":
    main()
