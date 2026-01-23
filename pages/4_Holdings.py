import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.streamlit_data import (
    load_watchlist,
    get_prices,
    get_fundamentals,
    get_scores,
    market_status
)
from src.nexus_ui import render_layout, render_kpi_card

def sidebar_content():
    st.write("Use the sidebar to upload portfolio files or edit the watchlist.")
    # In P12.4 we will refine this input mechanism
    
def context_content(selected_ticker, fundamentals, scores):
    """
    Research context for the selected holding.
    """
    if not selected_ticker:
        st.info("Select a holding to view research.")
        return

    st.subheader(f"{selected_ticker} Analysis")
    
    # Fundamentals
    if not fundamentals.empty and selected_ticker in fundamentals["ticker"].values:
        row = fundamentals[fundamentals["ticker"] == selected_ticker].iloc[0]
        st.markdown("**Fundamentals**")
        st.write(row.to_dict()) # Placeholder for better UI
        
    # Scores
    if not scores.empty and selected_ticker in scores["ticker"].values:
        s_row = scores[scores["ticker"] == selected_ticker].iloc[0]
        render_kpi_card("Composite", f"{s_row.get('composite_pct', 0):.1f}", "Percentile Rank")

def main_content():
    watchlist = load_watchlist()
    tickers = watchlist.get("tickers", [])
    
    st.write("### Holdings & Watchlist")
    
    if not tickers:
        st.warning("Watchlist is empty.")
        return

    # Data Loading
    _, market_state = market_status()
    prices, _ = get_prices(market_state, tickers)
    scores, _ = get_scores(tickers)
    fund, _ = get_fundamentals(tickers)
    
    # Main Table
    # For now, simplistic table. P12.4 will enhance with AgGrid.
    df = pd.DataFrame({"Ticker": tickers})
    # Merge price
    if not prices.empty:
        latest_prices = prices.sort_values("date").groupby("ticker").last()["adj_close"]
        df = df.merge(latest_prices, left_on="Ticker", right_index=True, how="left").rename(columns={"adj_close": "Price"})
        
    st.dataframe(df, use_container_width=True)
    
    # Selection (Temporary mechanism until AgGrid selection is ported)
    selected = st.selectbox("Select Holding for Detail", tickers)
    
    # Render Context (Inline for now if mobile, but Layout System handles it)
    # Actually, render_layout expects a function.
    # We need to bridge the selection state.
    st.session_state["nexus_selected_holding"] = selected


def page_main():
    st.set_page_config(page_title="Holdings", page_icon="📋", layout="wide")
    
    # We need to pre-load data to pass to context
    # This is a bit tricky with the callback structure of render_layout if data is needed in both.
    # Ideally, we load data first, then define closures.
    
    watchlist = load_watchlist()
    tickers = watchlist.get("tickers", [])
    _, market_state = market_status()
    # Optimization: Load only what's needed.
    # For Context, we need everything for the SELECTED ticker.
    
    # Let's perform a simple restructure:
    # 1. Sidebar
    # 2. Main (Table)
    # 3. Context (Detail)
    
    # Re-define functions with data closure
    def render_ctx():
        sel = st.session_state.get("nexus_selected_holding", tickers[0] if tickers else None)
        if sel:
            # Load specific data
            f_, _ = get_fundamentals([sel])
            s_, _ = get_scores([sel])
            context_content(sel, f_, s_)

    render_layout(
        "Holdings",
        sidebar_content,
        main_content,
        context_content_func=render_ctx
    )

if __name__ == "__main__":
    page_main()
