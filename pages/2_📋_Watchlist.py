from __future__ import annotations

import streamlit as st

from src.streamlit_data import (
    load_portfolio_cached,
    portfolio_cache_token,
    load_prices,
    load_scores,
    load_watchlist,
    market_status,
)
from src.streamlit_ui import build_status, render_guidance, render_header, render_portfolio_errors, render_sidebar
from src.guidance import explain_ticker_change

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
except Exception:  # pragma: no cover
    AgGrid = None
    GridOptionsBuilder = None

st.set_page_config(page_title="Watchlist", page_icon="📋", layout="wide")
render_sidebar()

watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
market_label, market_state = market_status()

scores = load_scores()
prices = load_prices(market_state)
portfolio = load_portfolio_cached(prices, watch_tickers, portfolio_cache_token())
status = build_status(prices, scores, watch_tickers, portfolio)
render_header("Watchlist", status)
render_portfolio_errors(portfolio)
if not scores.empty:
    scores = scores[scores["ticker"].isin(watch_tickers)]

if scores.empty:
    st.info("No watchlist data available.")
    st.stop()

scores = scores.copy()
cols = [
    "ticker",
    "Price",
    "composite_pct",
    "value_pct",
    "quality_pct",
    "momentum_pct",
    "PiotroskiF",
    "Volatility30d",
    "Sharpe1y",
    "industry",
]

scores = scores[cols].rename(columns={
    "Price": "Price",
    "composite_pct": "Composite %",
    "value_pct": "Value %",
    "quality_pct": "Quality %",
    "momentum_pct": "Momentum %",
    "PiotroskiF": "Piotroski F",
    "Volatility30d": "Vol 30d",
    "Sharpe1y": "Sharpe 1y",
    "industry": "Industry",
})

st.subheader("Interactive Table")
selected_ticker = None
if AgGrid and GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(scores)
    gb.configure_pagination(enabled=True, paginationPageSize=20)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    grid = AgGrid(scores, gridOptions=gb.build(), height=420)
    selected = grid.get("selected_rows", [])
    if selected:
        selected_ticker = selected[0].get("ticker")
else:
    st.dataframe(scores, use_container_width=True)

if selected_ticker:
    st.success(f"Selected: {selected_ticker}")
    if st.button("Open Stock Research"):
        st.session_state["selected_ticker"] = selected_ticker
        try:
            st.switch_page("pages/3_🔍_Stock_Research.py")
        except Exception:
            st.warning("Switch page failed. Use the Stock Research page from the sidebar.")

    summary = explain_ticker_change(selected_ticker, scores)
    mode = st.session_state.get("mode", "Simple")
    show = not scores.empty and selected_ticker in scores["ticker"].values
    render_guidance(summary, mode, show)

st.caption(f"Market: {market_label}")
