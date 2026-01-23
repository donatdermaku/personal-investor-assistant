from __future__ import annotations

import pandas as pd
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
from src.streamlit_ui import (
    build_status, 
    render_header, 
    render_portfolio_errors, 
    render_sidebar,
    ui_metric_card,
    ui_section_header
)

st.set_page_config(page_title="Personal Investor Assistant", page_icon="📈", layout="wide")

render_sidebar()

_, market_state = market_status()
watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
selected = st.sidebar.selectbox("Quick ticker", watch_tickers) if watch_tickers else None

scores, scores_meta = get_scores(watch_tickers)
prices, price_meta = get_prices(market_state, watch_tickers)
_, fundamentals_meta = get_fundamentals(watch_tickers)
portfolio = load_portfolio_cached(
    prices,
    watch_tickers,
    portfolio_cache_token(),
    source_override=st.session_state.get("portfolio_source"),
    uploads_active=st.session_state.get("uploads_active", False),
)

status = build_status(price_meta, fundamentals_meta, portfolio)
render_header("Personal Investor Assistant", status, {"Prices": price_meta, "Fundamentals": fundamentals_meta, "Scores": scores_meta})
render_portfolio_errors(portfolio)

# --- Top Metrics using Components ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    ui_metric_card("Watchlist size", len(watch_tickers), value_formatter="{:d}")
with col2:
    ui_metric_card("Universe size", int(scores["ticker"].nunique()) if not scores.empty else 0, value_formatter="{:d}")
with col3:
    if selected and not scores.empty and selected in scores["ticker"].values:
        pct = scores.set_index("ticker").loc[selected, "composite_pct"]
        ui_metric_card("Selected composite %", pct, value_formatter="{:.1f}")
    else:
        ui_metric_card("Selected composite %", None)
with col4:
    # Status dict already has formatted strings, so we pass as is
    ui_metric_card("Last update", status["last_run"], value_formatter="{}")


# --- Onboarding / Overview ---
ui_section_header("Overview")

# Check if portfolio is effectively empty/unset
if not st.session_state.get("uploads_active", False) and portfolio.source == "demo":
    # Onboarding Mode
    with st.container():
        st.info("Welcome! You are currently viewing **Demo Data**. Choose an option below to get started.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🚀 Quick Start")
            st.write("See what the app can do with pre-loaded demo data.")
            if st.button("Continue with Demo"):
                # Demo is default, so just maybe show a toast or nothing
                st.toast("Using Demo Data")
        
        with c2:
            st.subheader("📂 Upload Your Data")
            st.write("Upload your ledger or holdings snapshot via the sidebar to see your own portfolio.")
            st.caption("Supported formats: CSV (Ledger: date, ticker, amount | Snapshot: ticker, shares)")

else:
    # Standard Overview
    st.write(
        "Use the sidebar to navigate the dashboard and watchlist pages. "
        "This app reads from your existing DuckDB/Parquet outputs and does not modify the pipeline."
    )


if selected:
    ui_section_header("Quick View")
    if not scores.empty and selected in scores["ticker"].values:
        row = scores.set_index("ticker").loc[selected]
        q1, q2, q3, q4 = st.columns(4)
        
        with q1:
            price = row.get("Price")
            ui_metric_card("Price", price if pd.notna(price) else None)
        with q2:
            comp = row.get("composite_pct")
            ui_metric_card("Composite %", comp if pd.notna(comp) else None, value_formatter="{:.1f}")
        with q3:
            val = row.get("value_pct")
            ui_metric_card("Value %", val if pd.notna(val) else None, value_formatter="{:.1f}")
        with q4:
            mom = row.get("momentum_pct")
            ui_metric_card("Momentum %", mom if pd.notna(mom) else None, value_formatter="{:.1f}")

        if not prices.empty:
            series = prices[prices["ticker"] == selected].copy()
            if not series.empty:
                series["date"] = pd.to_datetime(series["date"])
                series = series.sort_values("date")
                st.line_chart(series.set_index("date")["adj_close"], height=220)
    else:
        st.info(f"Quick view selected: {selected}")
