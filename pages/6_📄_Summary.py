from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.streamlit_export import export_summary_html
from src.streamlit_ui import render_sidebar

st.set_page_config(page_title="Summary Export", page_icon="📄", layout="wide")
render_sidebar()

st.title("Summary Export")

sections = [
    ("Overview", "Informational summary for offline sharing."),
    ("Disclaimer", "This is informational and not financial advice."),
]

if st.button("Generate HTML summary"):
    path = Path("data/exports/summary.html")
    export_summary_html(path, sections)
    st.success(f"Saved: {path}")
