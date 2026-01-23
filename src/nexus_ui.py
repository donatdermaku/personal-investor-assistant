import streamlit as st
import datetime

# --- Design Tokens ---
COLOR_PRIMARY = "#0F172A"  # Dark Slate (Fintech trust)
COLOR_ACCENT = "#3B82F6"   # Blue (Action)
COLOR_BG_GRAY = "#F3F4F6"  # Neutral background
BORDER_RADIUS = "0.5rem"
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# --- Layout System ---
def set_fintech_style():
    """
    Injects CSS for a professional, neutral fintech aesthetic.
    """
    st.markdown(f"""
        <style>
        /* General Background */
        .stApp {{
            background-color: {COLOR_BG_GRAY};
            font-family: {FONT_FAMILY};
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }}
        
        /* metric labels */
        div[data-testid="stMetricLabel"] {{
            font-size: 0.8rem;
            color: #6B7280;
            text-transform: uppercase;
        }}
        
        /* metric values */
        div[data-testid="stMetricValue"] {{
            font-size: 1.5rem;
            color: {COLOR_PRIMARY};
            font-family: 'SF Mono', 'Roboto Mono', monospace;
        }}
        
        /* Remove default header decoration */
        header[data-testid="stHeader"] {{
            background-color: transparent;
        }}
        
        /* Adjust block padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Custom card styling wrapper */
        .nexus-card {{
            background: white;
            padding: 1.5rem;
            border-radius: {BORDER_RADIUS};
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_layout(
    curr_page: str,
    sidebar_content_func: callable,
    main_content_func: callable,
    context_content_func: callable | None = None
):
    """
    Implements the 3-zone layout:
    1. Sidebar (Navigation + Inputs)
    2. Main (Center Content)
    3. Context (Right Panel or Top Summary)
    """
    set_fintech_style()
    
    # 1. Sidebar
    with st.sidebar:
        st.markdown(f"### Nexus Analytics")
        st.markdown("---")
        sidebar_content_func()

    # 2. Header
    render_header(curr_page)

    # 3. Main & Context
    if context_content_func:
        c_main, c_context = st.columns([3, 1])
        with c_main:
            main_content_func()
        with c_context:
            st.markdown("#### Context")
            context_content_func()
    else:
        main_content_func()


def render_header(title: str):
    st.markdown(f"""
    <div style="border-bottom: 2px solid #E5E7EB; padding-bottom: 1rem; margin-bottom: 2rem;">
        <h1 style="font-family: {FONT_FAMILY}; color: {COLOR_PRIMARY}; margin: 0;">{title}</h1>
        <div style="color: #6B7280; font-size: 0.9rem;">Nexus Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = None, tooltip: str = None):
    """
    Clean, flat KPI card for fintech look.
    """
    tooltip_html = f'title="{tooltip}"' if tooltip else ""
    st.markdown(f"""
    <div {tooltip_html} style="
        background: white; 
        border: 1px solid #E5E7EB; 
        border-radius: {BORDER_RADIUS}; 
        padding: 1rem; 
        margin-bottom: 0.75rem;">
        <div style="color: #6B7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
        <div style="color: {COLOR_PRIMARY}; font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem;">{value}</div>
        {f'<div style="color: #9CA3AF; font-size: 0.75rem; margin-top: 0.25rem;">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)
