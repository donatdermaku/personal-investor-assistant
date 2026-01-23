from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from src.nexus_ui import render_layout, render_kpi_card
from src.streamlit_data import (
    get_fundamentals,
    get_prices,
    get_scores,
    load_portfolio_cached,
    portfolio_cache_token,
    load_watchlist,
    market_status,
    load_benchmark_prices
)
from src.portfolio import align_benchmark, compute_drawdown
from src.streamlit_ui import build_status, render_portfolio_errors

st.set_page_config(page_title="Overview - Nexus Analytics", page_icon="📈", layout="wide")

def sidebar_content():
    st.write("Settings and Input management will go here.")
    # Legacy controls for now
    st.session_state["portfolio_source"] = st.radio("Source", ["Auto", "Ledger", "Snapshot", "Demo"], index=0)

def context_content(status_dict):
    st.markdown("### Key Metrics")
    render_kpi_card("TWR (Strategy)", status_dict.get("twr_str", "--"), "Time-Weighted Return")
    render_kpi_card("MWR (Personal)", status_dict.get("mwr_str", "--"), "Money-Weighted Return")
    
    st.markdown("### System Status")
    st.info(f"Last Update: {status_dict.get('last_run', 'N/A')}")
    st.caption(f"Market: {status_dict.get('market_status', 'Unknown')}")
    
    # Export placeholder
    if st.button("Export Report"):
        st.toast("Export functionality coming in Phase 12.1 context integration.")

def main_content(portfolio, prices, bench_prices, benchmark):
    st.write("### Equity Curve")
    
    if portfolio.daily_values.empty:
        st.info("No portfolio data available.")
        return

    # Charting
    eq = portfolio.daily_values["value"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio", line=dict(color="#0F172A", width=2)))
    
    if not bench_prices.empty:
        bench_index = align_benchmark(bench_prices, eq)
        if not bench_index.empty:
            fig.add_trace(go.Scatter(x=bench_index.index, y=bench_index.values, name=benchmark, line=dict(color="#9CA3AF", width=1.5, dash="dot")))
            
    fig.update_layout(
        template="plotly_white", 
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Errors
    render_portfolio_errors(portfolio)

# --- Entry Point ---
def main():
    # Data Loading
    watchlist = load_watchlist()
    watch_tickers = watchlist.get("tickers", [])
    _, market_state = market_status()
    
    scores, scores_meta = get_scores(watch_tickers)
    prices, price_meta = get_prices(market_state, watch_tickers)
    _, fundamentals_meta = get_fundamentals(watch_tickers)
    
    portfolio = load_portfolio_cached(
        prices,
        watch_tickers,
        portfolio_cache_token(),
        source_override=st.session_state.get("portfolio_source", "Auto"),
        uploads_active=True, 
    )
    
    benchmark = "SPY"
    bench_prices = load_benchmark_prices(benchmark)
    
    status = build_status(price_meta, fundamentals_meta, portfolio)
    status["market_status"] = market_state
    
    # Render Layout
    # Closures to pass data
    render_layout(
        "Overview",
        sidebar_content,
        lambda: main_content(portfolio, prices, bench_prices, benchmark),
        lambda: context_content(status)
    )

if __name__ == "__main__":
    main()
