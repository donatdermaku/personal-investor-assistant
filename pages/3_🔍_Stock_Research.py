from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.streamlit_data import (
    load_fundamentals,
    load_news,
    load_portfolio_cached,
    portfolio_cache_token,
    load_prices,
    load_scores,
    load_universe,
    load_watchlist,
    market_status,
)
from src.streamlit_ui import build_status, render_guidance, render_header, render_portfolio_errors, render_sidebar
from src.guidance import explain_ticker_change

st.set_page_config(page_title="Stock Research", page_icon="🔍", layout="wide")
render_sidebar()

watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
market_label, market_state = market_status()

scores = load_scores()
prices = load_prices(market_state)
fundamentals = load_fundamentals()
universe = load_universe()
portfolio = load_portfolio_cached(prices, watch_tickers, portfolio_cache_token())
status = build_status(prices, scores, watch_tickers, portfolio)
render_header("Stock Research", status)
render_portfolio_errors(portfolio)

query = st.query_params.get("ticker")
preselect = query if query else st.session_state.get("selected_ticker")

selected = st.selectbox("Select ticker", watch_tickers, index=watch_tickers.index(preselect) if preselect in watch_tickers else 0)

if selected:
    st.session_state["selected_ticker"] = selected

row = scores[scores["ticker"] == selected].iloc[0] if not scores.empty and selected in scores["ticker"].values else None
vendor_ticker = selected
if not universe.empty and "vendor_ticker" in universe.columns:
    match = universe[universe["ticker"] == selected]
    if not match.empty:
        vendor_ticker = match.iloc[0]["vendor_ticker"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Price", f"{row['Price']:.2f}" if row is not None else "--")
with col2:
    st.metric("Composite %", f"{row['composite_pct']:.1f}" if row is not None else "--")
with col3:
    st.metric("Value %", f"{row['value_pct']:.1f}" if row is not None else "--")
with col4:
    st.metric("Quality %", f"{row['quality_pct']:.1f}" if row is not None else "--")

st.subheader("Price Chart")
if prices.empty:
    st.info("No price data available.")
else:
    df = prices[prices["ticker"] == selected].copy()
    if df.empty:
        st.info("No price history for this ticker.")
    else:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["sma20"] = df["adj_close"].rolling(20).mean()
        df["sma50"] = df["adj_close"].rolling(50).mean()

        delta = df["adj_close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace({0: np.nan})
        df["rsi14"] = 100 - (100 / (1 + rs))

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ))
        fig.add_trace(go.Scatter(x=df["date"], y=df["sma20"], name="SMA 20"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["sma50"], name="SMA 50"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df["date"], y=df["rsi14"], name="RSI 14"))
        rsi_fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(rsi_fig, use_container_width=True)

st.subheader("Fundamentals")
if fundamentals.empty:
    st.info("No fundamentals data available.")
else:
    f = fundamentals[fundamentals["ticker"] == selected].copy()
    if f.empty:
        st.info("No fundamentals history for this ticker.")
    else:
        f = f.sort_values("fiscal_end").tail(8)
        cols = [
            "fiscal_end",
            "Revenue",
            "NetIncome",
            "OperatingCF",
            "CapitalExpenditures",
            "TotalAssets",
            "TotalLiabilities",
        ]
        st.dataframe(f[cols], use_container_width=True)

st.subheader("SEC Filings")
if row is not None and pd.notna(row.get("cik")):
    cik = str(row.get("cik")).zfill(10)
    st.link_button("Open SEC filings", f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude")
    st.caption(f"Last filed: {row.get('filed')}")
else:
    st.info("No CIK available for this ticker.")

st.subheader("Peers")
if row is not None and row.get("industry"):
    industry = row.get("industry")
    peers = scores[scores["industry"] == industry].sort_values("composite_pct", ascending=False).head(10)
    st.dataframe(peers[["ticker", "composite_pct", "value_pct", "quality_pct", "momentum_pct"]], use_container_width=True)
else:
    st.info("No peer data available.")

st.subheader("News")
news_items = load_news(vendor_ticker)
if not news_items:
    st.info("News feed not available.")
else:
    for item in news_items[:5]:
        title = item.get("title")
        link = item.get("link")
        publisher = item.get("publisher")
        if title and link:
            st.markdown(f"- [{title}]({link}) — {publisher}")

st.caption(f"Market: {market_label}")

summary = explain_ticker_change(selected, scores)
mode = st.session_state.get("mode", "Simple")
show = not scores.empty and selected in scores["ticker"].values
render_guidance(summary, mode, show)
