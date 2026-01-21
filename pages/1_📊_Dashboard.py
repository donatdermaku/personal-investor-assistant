from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.streamlit_data import (
    load_benchmark_prices,
    load_portfolio_cached,
    load_prices,
    load_scores,
    load_scores_prior,
    load_watchlist,
    market_status,
    portfolio_cache_token,
)
from src.guidance import explain_portfolio
from src.intelligence import drawdown_intelligence, factor_tilts
from src.glossary import GLOSSARY
from src.streamlit_ui import build_status, render_guidance, render_header, render_portfolio_errors, render_sidebar
from src.portfolio import align_benchmark, compute_drawdown

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

render_sidebar()

watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
market_label, market_state = market_status()

scores = load_scores()
scores_prior = load_scores_prior()
prices = load_prices(market_state)
portfolio = load_portfolio_cached(prices, watch_tickers, portfolio_cache_token())
benchmark = st.session_state.get("benchmark", "SPY")
bench_prices = load_benchmark_prices(benchmark)

status = build_status(prices, scores, watch_tickers, portfolio)
render_header("Dashboard", status)
render_portfolio_errors(portfolio)

if portfolio.source == "snapshot":
    st.info("Holdings snapshot mode: performance history is limited to price history and does not include cashflows.")

summary = explain_portfolio(scores, portfolio.daily_returns, watch_tickers)
mode = st.session_state.get("mode", "Simple")
render_guidance(summary, mode, not portfolio.daily_returns.empty)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Equity Curve vs Benchmark")
    if prices.empty or not watch_tickers or portfolio.daily_values.empty:
        st.info("No price data available.")
    else:
        eq = portfolio.daily_values["value"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio"))
        if not bench_prices.empty:
            bench_index = align_benchmark(bench_prices, eq)
            if not bench_index.empty:
                fig.add_trace(go.Scatter(x=bench_index.index, y=bench_index.values, name=benchmark))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Allocation")
    weights = watchlist.get("weights", {}) or {}
    if weights:
        pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()))])
        pie.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No weights configured.")

st.subheader("Drawdown")
if portfolio.daily_values.empty:
    st.info("No drawdown data available.")
else:
    eq = portfolio.daily_values["value"]
    drawdown = compute_drawdown(eq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown"))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    dd_info = drawdown_intelligence(eq)
    if dd_info:
        st.caption(f"Max drawdown: {dd_info['max_drawdown']:.2%} | Current drawdown: {dd_info['current_drawdown']:.2%}")
        st.caption(GLOSSARY.get("Drawdown", ""))

st.subheader("Factor Tilts vs Benchmark")
if not scores.empty and watch_tickers:
    tilts = factor_tilts(scores, watch_tickers)
    if tilts.empty:
        st.info("No tilt data available.")
    else:
        st.dataframe(tilts, use_container_width=True)
        if mode.startswith("Pro"):
            st.caption("Tilt = portfolio mean percentile minus benchmark mean percentile (universe).")
            st.caption(GLOSSARY.get("Tilt", ""))
            st.download_button(
                "Download factor_tilts.csv",
                data=tilts.to_csv(index=False).encode("utf-8"),
                file_name="factor_tilts.csv",
                mime="text/csv",
            )
else:
    st.info("No tilt data available.")

st.caption(f"Market: {market_label}")
