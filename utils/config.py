import streamlit as st
import matplotlib.pyplot as plt

def setup_page_and_styling():
    st.set_page_config(
        page_title="LSTM Prediksi Energi Listrik",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.markdown("""
    <style>
        .stApp { background-color: #0f1117; }
        [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a2f3e; }
        .metric-card { background: linear-gradient(135deg, #1e2535, #252d42); border: 1px solid #2e3650; border-radius: 12px; padding: 18px 22px; text-align: center; margin-bottom: 10px; }
        .metric-label { color: #8899bb; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
        .metric-value { color: #e2e8f0; font-size: 26px; font-weight: 700; }
        .metric-sub   { color: #64748b; font-size: 12px; margin-top: 4px; }
        .metric-good  { color: #34d399; }
        .metric-mid   { color: #fbbf24; }
        .metric-bad   { color: #f87171; }
        .section-header { color: #60a5fa; font-size: 14px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid #2a2f3e; padding-bottom: 8px; margin: 20px 0 14px 0; }
        .badge-good  { background:#064e3b; color:#34d399; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:700; }
        .badge-mid   { background:#78350f; color:#fbbf24; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:700; }
        .badge-bad   { background:#7f1d1d; color:#f87171; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:700; }
        .info-banner { background: linear-gradient(90deg, #1e3a5f, #1e2d4a); border-left: 3px solid #3b82f6; border-radius: 6px; padding: 14px 18px; margin: 10px 0; color: #93c5fd; font-size: 13px; }
        hr { border-color: #2a2f3e; }
    </style>
    """, unsafe_allow_html=True)

    plt.rcParams.update({
        "figure.facecolor":  "#161b27",
        "axes.facecolor":    "#1a2035",
        "axes.edgecolor":    "#2e3650",
        "axes.labelcolor":   "#94a3b8",
        "xtick.color":       "#64748b",
        "ytick.color":       "#64748b",
        "text.color":        "#e2e8f0",
        "grid.color":        "#1e2a40",
        "grid.linewidth":    0.7,
        "legend.facecolor":  "#1e2535",
        "legend.edgecolor":  "#2e3650",
        "legend.labelcolor": "#e2e8f0",
        "font.family":       "DejaVu Sans",
    })