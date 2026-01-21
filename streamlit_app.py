from __future__ import annotations

import pandas as pd
import streamlit as st

from src.streamlit_data import (
    load_portfolio_cached,
    portfolio_cache_token,
    load_prices,
    load_scores,
    load_watchlist,
    market_status,
)
from src.streamlit_ui import build_status, render_header, render_portfolio_errors, render_sidebar

st.set_page_config(page_title="Personal Investor Assistant", page_icon="📈", layout="wide")

render_sidebar()

market_label, market_state = market_status()
watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
selected = st.sidebar.selectbox("Quick ticker", watch_tickers) if watch_tickers else None

scores = load_scores()
prices = load_prices(market_state)
portfolio = load_portfolio_cached(prices, watch_tickers, portfolio_cache_token())

status = build_status(prices, scores, watch_tickers, portfolio)
render_header("Personal Investor Assistant", status)
render_portfolio_errors(portfolio)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Watchlist size", len(watch_tickers))
with col2:
    st.metric("Universe size", int(scores["ticker"].nunique()) if not scores.empty else 0)
with col3:
    if selected and not scores.empty and selected in scores["ticker"].values:
        pct = scores.set_index("ticker").loc[selected, "composite_pct"]
        st.metric("Selected composite %", f"{pct:.1f}")
    else:
        st.metric("Selected composite %", "--")
with col4:
    st.metric("Last update", status["last_run"])

st.subheader("Overview")
st.write(
    "Use the sidebar to navigate the dashboard and watchlist pages. "
    "This app reads from your existing DuckDB/Parquet outputs and does not modify the pipeline."
)

if selected:
    st.subheader("Quick View")
    if not scores.empty and selected in scores["ticker"].values:
        row = scores.set_index("ticker").loc[selected]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Price", f"{row['Price']:.2f}" if pd.notna(row.get("Price")) else "--")
        q2.metric("Composite %", f"{row['composite_pct']:.1f}" if pd.notna(row.get("composite_pct")) else "--")
        q3.metric("Value %", f"{row['value_pct']:.1f}" if pd.notna(row.get("value_pct")) else "--")
        q4.metric("Momentum %", f"{row['momentum_pct']:.1f}" if pd.notna(row.get("momentum_pct")) else "--")

        if not prices.empty:
            series = prices[prices["ticker"] == selected].copy()
            if not series.empty:
                series["date"] = pd.to_datetime(series["date"])
                series = series.sort_values("date")
                st.line_chart(series.set_index("date")["adj_close"], height=220)
    else:
        st.info(f"Quick view selected: {selected}")
