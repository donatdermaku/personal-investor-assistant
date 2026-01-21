from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils_io import ROOT


def init_session_state() -> None:
    defaults = {
        "mode": "Simple",
        "portfolio_source": "Auto",
        "benchmark": "SPY",
        "base_currency": "USD",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    init_session_state()
    st.sidebar.header("Settings")
    st.sidebar.radio("Mode", ["Simple", "Pro (Quant)"], key="mode")
    st.sidebar.selectbox("Base currency", ["USD"], key="base_currency")
    st.sidebar.text_input("Benchmark", key="benchmark")

    st.sidebar.header("Portfolio Input")
    ledger_upload = st.sidebar.file_uploader("Upload ledger CSV", type=["csv"], key="ledger_upload")
    snapshot_upload = st.sidebar.file_uploader("Upload holdings snapshot CSV", type=["csv"], key="snapshot_upload")

    uploads_dir = ROOT / "data" / "user_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    if ledger_upload is not None:
        dest = uploads_dir / "transactions.csv"
        dest.write_bytes(ledger_upload.getvalue())
        st.sidebar.success("Ledger saved: data/user_uploads/transactions.csv")

    if snapshot_upload is not None:
        dest = uploads_dir / "holdings.csv"
        dest.write_bytes(snapshot_upload.getvalue())
        st.sidebar.success("Snapshot saved: data/user_uploads/holdings.csv")

    st.sidebar.caption("Source precedence: Ledger → Snapshot → Demo")

    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()


def render_header(title: str, status: dict) -> None:
    st.title(title)
    cols = st.columns(4)
    cols[0].metric("Last pipeline run", status.get("last_run", "--"))
    cols[1].metric("Price coverage", status.get("price_coverage", "--"))
    cols[2].metric("Fundamentals coverage", status.get("fundamentals_coverage", "--"))
    cols[3].metric("Portfolio source", status.get("portfolio_source", "--"))
    st.caption("This is informational and not financial advice.")


def build_status(prices, scores, watch_tickers, portfolio) -> dict:
    last_update = None
    if prices is not None and not prices.empty:
        last_update = pd.to_datetime(prices["date"]).max()

    price_coverage = "--"
    fund_coverage = "--"
    if watch_tickers and prices is not None and not prices.empty:
        covered = prices["ticker"].isin(watch_tickers).groupby(prices["ticker"]).any().sum()
        price_coverage = f"{covered}/{len(watch_tickers)}"
    if scores is not None and not scores.empty and watch_tickers:
        has_fund = scores[scores["ticker"].isin(watch_tickers)]["has_fundamentals"].fillna(False).mean()
        fund_coverage = f"{has_fund * 100:.0f}%"

    return {
        "last_run": last_update.strftime("%Y-%m-%d") if last_update is not None else "--",
        "price_coverage": price_coverage,
        "fundamentals_coverage": fund_coverage,
        "portfolio_source": portfolio.source.capitalize() if portfolio else "--",
    }


def render_portfolio_errors(portfolio) -> None:
    if portfolio and portfolio.errors:
        st.warning("Portfolio input issues:\n- " + "\n- ".join(portfolio.errors))


def _render_list(title: str, items: list[str]) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    st.markdown("\n".join([f"- {item}" for item in items]))


def render_guidance(summary, mode: str, show: bool) -> None:
    st.subheader("Guidance")
    if not show:
        st.info("Guidance unavailable until data loads.")
        return

    if mode.startswith("Pro"):
        _render_list("What changed", summary.what_changed)
        _render_list("Why", summary.why)
        _render_list("Risks", summary.risk_warnings)
        _render_list("What next", summary.next_steps)
    else:
        _render_list("What changed", summary.what_changed)
        if summary.risk_warnings:
            st.warning(" / ".join(summary.risk_warnings))
