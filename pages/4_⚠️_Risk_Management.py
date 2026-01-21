from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.streamlit_data import (
    get_fundamentals,
    get_prices,
    get_scores,
    load_portfolio_cached,
    portfolio_cache_token,
    load_watchlist,
    market_status,
)
from src.streamlit_ui import build_status, render_guidance, render_header, render_portfolio_errors, render_sidebar
from src.guidance import risk_warnings
from src.glossary import GLOSSARY
from src.intelligence import component_risk

st.set_page_config(page_title="Risk Management", page_icon="⚠️", layout="wide")
render_sidebar()

watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
market_label, market_state = market_status()
prices, price_meta = get_prices(market_state, watch_tickers)
scores, scores_meta = get_scores(watch_tickers)
_, fundamentals_meta = get_fundamentals(watch_tickers)
portfolio = load_portfolio_cached(
    prices,
    watch_tickers,
    portfolio_cache_token(),
    source_override=st.session_state.get("portfolio_source"),
    uploads_active=st.session_state.get("uploads_active", False),
)
status = build_status(price_meta, fundamentals_meta, portfolio)
render_header("Risk Management", status, {"Prices": price_meta, "Fundamentals": fundamentals_meta, "Scores": scores_meta})
render_portfolio_errors(portfolio)

if portfolio.source == "snapshot":
    st.info("Holdings snapshot mode: risk metrics use price history without cashflow effects.")

warnings = risk_warnings(portfolio.daily_returns, scores, watch_tickers)
summary = type("Summary", (), {"what_changed": [], "why": [], "risk_warnings": warnings, "next_steps": []})()
mode = st.session_state.get("mode", "Simple")
render_guidance(summary, mode, not portfolio.daily_returns.empty or not scores.empty)

if prices.empty or not watch_tickers:
    st.info("No price data available for risk analytics.")
    st.stop()

prices = prices[prices["ticker"].isin(watch_tickers)].copy()
prices["date"] = pd.to_datetime(prices["date"])
wide = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
wide = wide.ffill().dropna(how="all")

if wide.empty:
    st.info("Insufficient price history.")
    st.stop()

returns = wide.pct_change().dropna()
portfolio_returns = portfolio.daily_returns

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Correlation Matrix")
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("VaR / CVaR")
    conf = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
    port_ret = portfolio_returns if not portfolio_returns.empty else returns.mean(axis=1)
    var = np.quantile(port_ret, 1 - conf)
    cvar = port_ret[port_ret <= var].mean() if not port_ret.empty else np.nan
    st.metric("VaR (daily)", f"{var:.2%}")
    st.metric("CVaR (daily)", f"{cvar:.2%}")
    st.caption(f"{GLOSSARY.get('VaR', '')} {GLOSSARY.get('CVaR', '')}")

st.subheader("Stress Test")
shock = st.slider("Market shock", -0.3, 0.3, -0.1, 0.01)
latest = wide.iloc[-1]
shock_value = latest * (1 + shock)
shock_df = pd.DataFrame({"Current": latest, "Shocked": shock_value})
shock_df["Impact %"] = (shock_df["Shocked"] / shock_df["Current"] - 1) * 100
st.dataframe(shock_df, use_container_width=True)

st.subheader("Position Sizing Calculator")
col_a, col_b, col_c = st.columns(3)
with col_a:
    portfolio_value = st.number_input("Portfolio value", min_value=1000.0, value=100000.0, step=1000.0)
with col_b:
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, value=1.0, step=0.1)
with col_c:
    stop_loss_pct = st.number_input("Stop loss (%)", min_value=0.1, value=5.0, step=0.5)

selected = st.selectbox("Select ticker", watch_tickers)
price = latest.get(selected)
if price and price == price:
    risk_amount = portfolio_value * (risk_pct / 100)
    shares = risk_amount / (price * (stop_loss_pct / 100))
    st.success(f"Estimated shares for {selected}: {shares:,.0f}")
else:
    st.info("Price not available for sizing.")

st.caption(f"Market: {market_label}")

st.subheader("Component Risk (Pro)")
mode = st.session_state.get("mode", "Simple")
if mode.startswith("Pro"):
    weights = watchlist.get("weights", {}) or {}
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    if w.sum() > 0:
        w = w / w.sum()
    contrib = component_risk(returns, w)
    if contrib.empty:
        st.info("No component risk data available.")
    else:
        st.dataframe(contrib, use_container_width=True)
