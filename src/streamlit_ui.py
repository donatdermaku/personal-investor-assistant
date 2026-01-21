from __future__ import annotations

import json
import streamlit as st

from src.streamlit_data import CoverageMeta
from src.utils_io import ROOT


def init_session_state() -> None:
    defaults = {
        "mode": "Simple",
        "portfolio_source": "Auto",
        "benchmark": "SPY",
        "base_currency": "USD",
        "uploads_active": False,
    }
    saved = _load_ui_state()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = saved.get(key, value)


def _load_ui_state() -> dict:
    path = ROOT / "data" / "user_uploads" / "ui_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_ui_state() -> None:
    path = ROOT / "data" / "user_uploads" / "ui_state.json"
    payload = {
        "mode": st.session_state.get("mode"),
        "portfolio_source": st.session_state.get("portfolio_source"),
        "benchmark": st.session_state.get("benchmark"),
        "base_currency": st.session_state.get("base_currency"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def render_sidebar() -> None:
    init_session_state()
    if "portfolio_source_pending" in st.session_state:
        st.session_state["portfolio_source"] = st.session_state.pop("portfolio_source_pending")
    st.sidebar.header("Settings")
    st.sidebar.radio("Mode", ["Simple", "Pro (Quant)"], key="mode", on_change=_save_ui_state)
    st.sidebar.selectbox("Base currency", ["USD"], key="base_currency", on_change=_save_ui_state)
    st.sidebar.text_input("Benchmark", key="benchmark", on_change=_save_ui_state)

    st.sidebar.header("Portfolio Input")
    st.sidebar.selectbox("Portfolio source", ["Auto", "Ledger", "Snapshot", "Demo"], key="portfolio_source", on_change=_save_ui_state)
    ledger_upload = st.sidebar.file_uploader("Upload ledger CSV", type=["csv"], key="ledger_upload")
    snapshot_upload = st.sidebar.file_uploader("Upload holdings snapshot CSV", type=["csv"], key="snapshot_upload")

    uploads_dir = ROOT / "data" / "user_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    if ledger_upload is not None:
        dest = uploads_dir / "transactions.csv"
        dest.write_bytes(ledger_upload.getvalue())
        st.sidebar.success("Ledger saved: data/user_uploads/transactions.csv")
        st.session_state["portfolio_source_pending"] = "Ledger"
        st.session_state["uploads_active"] = True

    if snapshot_upload is not None:
        dest = uploads_dir / "holdings.csv"
        dest.write_bytes(snapshot_upload.getvalue())
        st.sidebar.success("Snapshot saved: data/user_uploads/holdings.csv")
        st.session_state["portfolio_source_pending"] = "Snapshot"
        st.session_state["uploads_active"] = True

    st.sidebar.caption("Source precedence: Ledger → Snapshot → Demo")
    st.sidebar.caption(f"Active mode: {st.session_state.get('mode', '--')}")
    st.sidebar.caption(f"Active source: {st.session_state.get('portfolio_source', '--')}")
    st.sidebar.caption(f"Uploads active: {st.session_state.get('uploads_active', False)}")

    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()


def render_header(title: str, status: dict, coverage_map: dict[str, CoverageMeta] | None = None) -> None:
    st.title(title)
    cols = st.columns(4)
    cols[0].metric("Last pipeline run", status.get("last_run", "--"))
    cols[1].metric("Price coverage", status.get("price_coverage", "--"))
    cols[2].metric("Fundamentals coverage", status.get("fundamentals_coverage", "--"))
    cols[3].metric("Portfolio source", status.get("portfolio_source", "--"))
    st.caption("This is informational and not financial advice.")

    if coverage_map:
        _render_missingness(coverage_map)


def build_status(price_meta: CoverageMeta, fundamentals_meta: CoverageMeta, portfolio) -> dict:
    last_update = price_meta.last_date or "--"
    price_coverage = _format_coverage(price_meta)
    fund_coverage = _format_coverage(fundamentals_meta)
    return {
        "last_run": last_update,
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


def _format_coverage(meta: CoverageMeta) -> str:
    if meta.total <= 0:
        return "--"
    return f"{meta.covered}/{meta.total}"


def _render_missingness(coverage_map: dict[str, CoverageMeta]) -> None:
    issues = False
    for meta in coverage_map.values():
        if meta.reasons or meta.missing_tickers:
            issues = True
            break
    if not issues:
        return

    with st.expander("Why missing?"):
        for label, meta in coverage_map.items():
            if not meta.reasons and not meta.missing_tickers and not meta.notes:
                continue
            st.markdown(f"**{label}**")
            if meta.reasons:
                st.markdown("Reasons: " + ", ".join(sorted(meta.reasons.keys())))
            if meta.notes:
                for note in meta.notes:
                    st.markdown(f"- {note}")
            if meta.missing_tickers:
                preview = meta.missing_tickers[:20]
                more = ""
                if len(meta.missing_tickers) > len(preview):
                    more = f" (+{len(meta.missing_tickers) - len(preview)} more)"
                st.markdown("Missing tickers: " + ", ".join(preview) + more)
