from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.streamlit_data import (
    get_fundamentals,
    get_prices,
    get_scores,
    load_benchmark_prices,
    load_portfolio_cached,
    portfolio_cache_token,
    load_watchlist,
    market_status,
)
from src.streamlit_ui import (
    build_status, 
    render_guidance, 
    render_header, 
    render_portfolio_errors, 
    render_sidebar,
    ui_metric_card,
    ui_section_header,
    ui_empty_state
)

# ... (rest of imports)

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
render_sidebar()

# ... (data loading)
watchlist = load_watchlist()
watch_tickers = watchlist.get("tickers", [])
weights = watchlist.get("weights", {}) or {}
market_label, market_state = market_status()

# Restoring data loading calls
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
render_header("Performance", status, {"Prices": price_meta, "Fundamentals": fundamentals_meta, "Scores": scores_meta})
render_portfolio_errors(portfolio)

benchmark = st.session_state.get("benchmark", "SPY")
bench_prices = load_benchmark_prices(benchmark)
mode = st.session_state.get("mode", "Simple")
is_pro = mode.startswith("Pro")

if portfolio.source == "snapshot":
    ui_empty_state("Snapshot Mode", "Holdings snapshot mode: performance history is limited to price history and does not include cashflows.", icon="📸")

if prices.empty or not watch_tickers or portfolio.daily_values.empty:
    ui_empty_state("No Data", "No price data available for performance analytics.", icon="📉")
    st.stop()

ui_section_header("Return Summary")
col_a, col_b = st.columns(2)
with col_a:
    if portfolio.twr is not None:
        ui_metric_card("TWR — Strategy Return", portfolio.twr, value_formatter="{:.2%}", help_text=GLOSSARY.get("TWR", "") + " Neutralizes external cashflows.")
    else:
        ui_metric_card("TWR — Strategy Return", None, help_text="Time-weighted return neutralizes external cashflows.")
with col_b:
    if is_pro:
        if portfolio.mwr is not None:
             ui_metric_card("MWR — Your Personal Return", portfolio.mwr, value_formatter="{:.2%}", help_text=GLOSSARY.get("MWR", ""))
        else:
            ui_metric_card("MWR — Your Personal Return", None, help_text="MWR unavailable (insufficient cashflows or convergence failure).")
    else:
        ui_metric_card("MWR — Your Personal Return", None, help_text="Switch to Pro mode to view money-weighted return.")

summary = explain_portfolio(scores, portfolio.daily_returns, watch_tickers)
mode = st.session_state.get("mode", "Simple")
render_guidance(summary, mode, not portfolio.daily_returns.empty)

# ... (chart prep code unchanged)

if wide.empty:
    ui_empty_state("Insufficient Data", "Insufficient price history.", icon="📉")
    st.stop()

# ... (calculation code unchanged)

col1, col2 = st.columns([2, 1])
with col1:
    ui_section_header("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_index.index, y=portfolio_index.values, name="Portfolio"))
    if not bench_prices.empty:
        bench_index = align_benchmark(bench_prices, portfolio_index)
        if not bench_index.empty:
            fig.add_trace(go.Scatter(x=bench_index.index, y=bench_index.values, name=benchmark))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    ui_section_header("Attribution (30d)")
    recent = returns.tail(30)
    contrib = recent.mean() * w
    contrib = contrib.sort_values(ascending=False)
    fig = go.Figure(data=[go.Bar(x=contrib.index, y=contrib.values)])
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

ui_section_header("Monthly Heatmap")
monthly = compute_monthly_returns(portfolio_returns)
if not monthly.empty:
    heat = monthly.to_frame("return")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="return").fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(m) for m in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="RdYlGn",
        zmin=-0.2,
        zmax=0.2,
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    ui_empty_state("No Data", "Insufficient data for heatmap.", icon="📅")

ui_section_header("Rolling Risk Metrics")
window = st.slider("Rolling window (days)", 30, 252, 60, 10)
roll_mean = portfolio_returns.rolling(window).mean()
roll_vol = portfolio_returns.rolling(window).std()
roll_sharpe = (roll_mean / roll_vol.replace({0: np.nan})) * np.sqrt(252)

fig = go.Figure()
fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, name="Volatility"))
fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="Sharpe"))
fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

ui_section_header("Component Risk")
if not portfolio_returns.empty and not returns.empty:
    contrib = component_risk(returns, w)
    if contrib.empty:
        ui_empty_state("No Data", "No component risk data available.", icon="⚠️")
    else:
        st.dataframe(contrib, use_container_width=True)
        if is_pro:
            st.download_button(
                "Download component_risk.csv",
                data=contrib.to_csv(index=False).encode("utf-8"),
                file_name="component_risk.csv",
                mime="text/csv",
            )
else:
    ui_empty_state("No Data", "No component risk data available.", icon="⚠️")

ui_section_header("Transaction History")
st.caption("Upload a broker CSV from the sidebar. The normalized ledger appears here.")

try:
    tx_path = ROOT / "data" / "user_uploads" / "transactions.csv"
    if tx_path.exists():
        tx = pd.read_csv(tx_path)
        st.dataframe(tx, use_container_width=True)
    else:
        ui_empty_state("No Ledger", "No ledger uploaded.", icon="📂")
except Exception:
    ui_empty_state("Error", "Unable to read transactions file.", icon="⚠️")

st.caption(f"Market: {market_label}")

if is_pro:

    st.subheader("Exports (Pro)")
    
    # Manifest integration for exports
    manifest = st.session_state.get("run_manifest")
    
    # We create a temporary structure for summary JSON that includes the manifest
    # Since we can't easily hook into the download button's internal callback to generate on fly with arguments,
    # we pre-generate the summary JSON with manifest if needed.
    
    # Note: For strict correctness, we'd refactor export_summary_json to return string/bytes 
    # instead of writing to file, but we'll stick to the existing pattern or adapt.
    # Actually, export_summary_json writes to a path, which isn't ideal for st.download_button.
    # The existing code doesn't seem to use export_summary_json for a button?
    # Ah, I see the existing code only has CSV downloads.
    
    # Let's ADD a summary JSON export which is the point of this feature.
    
    from src.streamlit_export import export_summary_json
    import json
    
    # We will compute the JSON string in memory for the download button
    summary_payload = {
        "source": portfolio.source,
        "twr": portfolio.twr,
        "mwr": portfolio.mwr,
        "final_value": portfolio.daily_values["value"].iloc[-1] if not portfolio.daily_values.empty else None,
        "last_date": portfolio.daily_values.index[-1].strftime("%Y-%m-%d") if not portfolio.daily_values.empty else None,
        "run_id": manifest.run_id if manifest else None,
        "input_hash": manifest.input_hash if manifest else None,
        "timestamp": manifest.timestamp if manifest else None,
        "errors": portfolio.errors,
    }
    
    st.download_button(
        "Download summary.json (Run Metadata)",
        data=json.dumps(summary_payload, indent=2).encode("utf-8"),
        file_name=f"summary_{manifest.run_id[:8] if manifest else 'run'}.json",
        mime="application/json",
    )

    st.download_button(
        "Download portfolio_daily_values.csv",
        data=portfolio.daily_values.to_csv().encode("utf-8"),
        file_name="portfolio_daily_values.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download portfolio_daily_returns.csv",
        data=portfolio.daily_returns.to_csv().encode("utf-8"),
        file_name="portfolio_daily_returns.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cashflows.csv",
        data=portfolio.cashflows.to_csv().encode("utf-8"),
        file_name="cashflows.csv",
        mime="text/csv",
    )

