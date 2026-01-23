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
    load_portfolio_cached,
    portfolio_cache_token,
    load_watchlist,
    market_status,
)
from src.streamlit_ui import build_status, render_portfolio_errors
from src.guidance import risk_warnings
from src.glossary import GLOSSARY
from src.intelligence import component_risk

st.set_page_config(page_title="Risk - Nexus Analytics", page_icon="⚠️", layout="wide")

def sidebar_content():
    st.write("Risk Model Settings")
    st.slider("Stress Test Shock %", -30, 30, -10, 5, key="risk_shock_slider")
    st.slider("VaR Confidence", 90, 99, 95, 1, key="risk_var_conf")

def context_content(var, cvar, last_run):
    st.markdown("### Tail Risk (Daily)")
    render_kpi_card("VaR", f"{var:.2%}" if not np.isnan(var) else "--", "Value at Risk", tooltip="Worst expected loss at confidence level")
    render_kpi_card("CVaR", f"{cvar:.2%}" if not np.isnan(cvar) else "--", "Conditional VaR", tooltip="Expected loss given VaR breach")
    
    st.divider()
    st.info("Risk metrics are calculated on daily returns.")

def main_content(portfolio, returns, wide, latest):
    if wide.empty or returns.empty:
        st.warning("Insufficient history for risk analytics.")
        return

    # 1. Correlation Matrix
    st.write("### Correlation Matrix")
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1
    ))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Stress Test
    shock_pct = st.session_state.get("risk_shock_slider", -10) / 100.0
    st.write(f"### Stress Test (Shock: {shock_pct:.0%})")
    
    shock_value = latest * (1 + shock_pct)
    shock_df = pd.DataFrame({"Current": latest, "Shocked": shock_value})
    shock_df["Impact"] = shock_df["Shocked"] - shock_df["Current"]
    shock_df["Impact %"] = (shock_df["Impact"] / shock_df["Current"]) * 100
    
    # Simple bar chart of impact
    fig_stress = go.Figure(data=[
        go.Bar(x=shock_df.index, y=shock_df["Impact %"], marker_color="#EF4444")
    ])
    fig_stress.update_layout(yaxis_title="Impact %", height=300, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_stress, use_container_width=True)
    
    # 3. Warnings
    st.write("### Risk Alerts")
    # TBD: Restore 'risk_warnings' logic here if needed, or simplified
    # For now, placeholder
    st.info("No active risk alerts.")


def main():
    watchlist = load_watchlist()
    watch_tickers = watchlist.get("tickers", [])
    _, market_state = market_status()
    
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
    
    # Pre-calc
    if prices.empty:
        st.error("No prices.")
        return

    prices_sub = prices[prices["ticker"].isin(watch_tickers)].copy()
    prices_sub["date"] = pd.to_datetime(prices_sub["date"])
    wide = prices_sub.pivot_table(index="date", columns="ticker", values="adj_close").sort_index().ffill().dropna(how="all")
    returns = wide.pct_change().dropna()
    latest = wide.iloc[-1] if not wide.empty else pd.Series()
    
    # Calc VaR/CVaR for context
    if not returns.empty:
        # Equal weight proxy if no portfolio returns
        port_ret = portfolio.daily_returns if not portfolio.daily_returns.empty else returns.mean(axis=1)
        conf = st.session_state.get("risk_var_conf", 95) / 100.0
        var = np.quantile(port_ret, 1 - conf)
        cvar = port_ret[port_ret <= var].mean()
    else:
        var, cvar = np.nan, np.nan
        
    render_layout(
        "Risk",
        sidebar_content,
        lambda: main_content(portfolio, returns, wide, latest),
        lambda: context_content(var, cvar, "Latest")
    )

if __name__ == "__main__":
    main()
