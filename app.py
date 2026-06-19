"""
MetaMavericks — Unified AI/ML Model Dashboard
================================================
A premium Streamlit application integrating three ML models:
  T1: Boston Housing Price Prediction (Regression)
  T2: NEXUS-RT v1 — Superstore Sales Analytics (Dual-Head Classification)
  T3: NLP — BERT Sentiment + GPT-2 Text Generation
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Krisha Patel — AI Model Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Base Directory ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Session State Defaults ────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠  Home"
if "bert_loaded" not in st.session_state:
    st.session_state.bert_loaded = False
if "gpt2_loaded" not in st.session_state:
    st.session_state.gpt2_loaded = False

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium Dark Theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Import Google Font ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global Overrides ────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #13132B 50%, #0F0F1A 100%);
}

/* ── Hide default Streamlit elements ─────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Sidebar Styling ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132B 0%, #1A1A35 100%) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.15);
    min-width: 310px !important;
    max-width: 310px !important;
    width: 310px !important;
    transform: none !important;
}

/* Hide the collapse / close button so sidebar stays open permanently */
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
    pointer-events: none !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    color: #A0AEC0 !important;
    font-weight: 500;
}

/* ── Glass Card ──────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(26, 26, 46, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.35);
    box-shadow: 0 12px 48px rgba(124, 58, 237, 0.1);
    transform: translateY(-2px);
}

/* ── Hero Section ────────────────────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(168, 85, 247, 0.2));
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 100px;
    padding: 8px 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #A78BFA;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0 0%, #A78BFA 50%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 16px;
    line-height: 1.15;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #94A3B8;
    max-width: 650px;
    margin: 0 auto 40px;
    line-height: 1.7;
    font-weight: 400;
}

/* ── Stat Cards ──────────────────────────────────────────────────────────── */
.stat-card {
    background: rgba(26, 26, 46, 0.5);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-card:hover {
    border-color: rgba(124, 58, 237, 0.3);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.08);
}

.stat-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}

.stat-value {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #C4B5FD, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 0.82rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ── Model Cards ─────────────────────────────────────────────────────────── */
.model-card {
    background: rgba(26, 26, 46, 0.5);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 16px;
    padding: 30px 24px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
    height: 100%;
}

.model-card:hover {
    border-color: rgba(124, 58, 237, 0.4);
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(124, 58, 237, 0.12);
}

.model-card-icon {
    font-size: 2.4rem;
    margin-bottom: 14px;
}

.model-card-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 8px;
}

.model-card-desc {
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.6;
    margin-bottom: 16px;
}

.model-card-metric {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(168, 85, 247, 0.1));
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #A78BFA;
}

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}

.section-subheader {
    font-size: 0.95rem;
    color: #64748B;
    margin-bottom: 30px;
    font-weight: 400;
}

/* ── Page Header ─────────────────────────────────────────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}

.page-header-icon {
    font-size: 2.2rem;
}

.page-header-text {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-desc {
    font-size: 0.95rem;
    color: #64748B;
    margin-bottom: 32px;
    padding-left: 4px;
}

/* ── Result Cards ────────────────────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(168, 85, 247, 0.04));
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.result-value {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #C4B5FD, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-label {
    font-size: 0.9rem;
    color: #94A3B8;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Sentiment Badges ────────────────────────────────────────────────────── */
.sentiment-positive {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(34, 197, 94, 0.06));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.sentiment-negative {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.06));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.sentiment-label-pos {
    font-size: 2rem;
    font-weight: 800;
    color: #22C55E;
}

.sentiment-label-neg {
    font-size: 2rem;
    font-weight: 800;
    color: #EF4444;
}

/* ── Generation Box ──────────────────────────────────────────────────────── */
.gen-box {
    background: rgba(26, 26, 46, 0.6);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 12px;
    padding: 24px;
    font-size: 0.95rem;
    color: #CBD5E1;
    line-height: 1.8;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* ── Profit Badge ────────────────────────────────────────────────────────── */
.profit-yes {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(34, 197, 94, 0.06));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.profit-no {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.06));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.3), transparent);
    margin: 32px 0;
    border: none;
}

/* ── Streamlit Widget Overrides ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
    box-shadow: 0 6px 25px rgba(124, 58, 237, 0.45) !important;
    transform: translateY(-1px);
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: rgba(26, 26, 46, 0.8) !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus,
.stTextArea textarea:focus {
    border-color: rgba(124, 58, 237, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
}

.stSlider [data-testid="stThumbValue"] {
    color: #A78BFA !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(26, 26, 46, 0.3);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.2) !important;
    color: #C4B5FD !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(26, 26, 46, 0.5) !important;
    border: 1px solid rgba(124, 58, 237, 0.12) !important;
    border-radius: 10px !important;
    color: #C4B5FD !important;
    font-weight: 600 !important;
}

/* ── Animated Pulse ──────────────────────────────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px rgba(124, 58, 237, 0.2); }
    50% { box-shadow: 0 0 30px rgba(124, 58, 237, 0.4); }
}

.pulse-border {
    animation: pulse-glow 3s ease-in-out infinite;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 15, 26, 0.5);
}
::-webkit-scrollbar-thumb {
    background: rgba(124, 58, 237, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(124, 58, 237, 0.5);
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 24px;
    color: #475569;
    font-size: 0.8rem;
    border-top: 1px solid rgba(124, 58, 237, 0.08);
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def _on_page_change():
    """Callback to persist page selection in session state."""
    st.session_state.current_page = st.session_state._nav_radio

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size: 2.4rem; margin-bottom: 6px;">🧠</div>
        <div style="font-size: 1.3rem; font-weight: 800; 
             background: linear-gradient(135deg, #C4B5FD, #7C3AED);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text;">Krisha Patel</div>
        <div style="font-size: 0.72rem; color: #64748B; letter-spacing: 2px; 
             text-transform: uppercase; margin-top: 4px;">AI Model Hub</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    _nav_options = ["🏠  Home", "🏘️  Housing Predictor", "📊  Sales Analytics", "💬  NLP Studio"]
    _default_idx = _nav_options.index(st.session_state.current_page) if st.session_state.current_page in _nav_options else 0

    st.radio(
        "Navigation",
        _nav_options,
        index=_default_idx,
        key="_nav_radio",
        on_change=_on_page_change,
        label_visibility="collapsed",
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 12px; background: rgba(124,58,237,0.06); 
         border-radius: 10px; border: 1px solid rgba(124,58,237,0.1);">
        <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; 
             letter-spacing: 1px; font-weight: 600; margin-bottom: 8px;">Model Status</div>
        <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 4px;">
            <span style="color: #22C55E;">●</span> &nbsp;T1 — Housing Net
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 4px;">
            <span style="color: #22C55E;">●</span> &nbsp;T2 — NEXUS-RT v1
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8;">
            <span style="color: #22C55E;">●</span> &nbsp;T3 — BERT + GPT-2
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-footer">
        Built with ❤️ by Krisha Patel<br/>
        Streamlit · PyTorch · Transformers
    </div>
    """, unsafe_allow_html=True)

# Read the page from session state (survives sidebar collapse)
page = st.session_state.current_page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    # ── Hero ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">✦ &nbsp;AI / ML Internship Portfolio</div>
        <div class="hero-title">Unified AI Model Hub</div>
        <div class="hero-subtitle">
            Three production-grade deep learning models — Regression, Multi-Task Classification 
            and NLP — unified into a single interactive experience.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats Row ──
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("🧬", "3", "Models Deployed"),
        ("⚡", "174K+", "Total Parameters"),
        ("🎯", "99.75%", "Peak Accuracy"),
        ("📐", "4", "Task Types"),
    ]
    for col, (icon, value, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Model Cards ──
    st.markdown('<div class="section-header">Deployed Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Click any model in the sidebar to explore its capabilities</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="model-card">
            <div class="model-card-icon">🏘️</div>
            <div class="model-card-title">Boston Housing Predictor</div>
            <div class="model-card-desc">
                Lightweight residual MLP that predicts median home values from 13 neighborhood 
                features. Uses log-transform, feature engineering, and OneCycleLR for fast convergence.
            </div>
            <div class="model-card-metric">R² > 0.80 · Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="model-card">
            <div class="model-card-icon">📊</div>
            <div class="model-card-title">NEXUS-RT v1 — Sales Analytics</div>
            <div class="model-card-desc">
                Dual-head residual MLP with 46 engineered features for simultaneous profitability 
                prediction and 17-class sub-category classification on retail data.
            </div>
            <div class="model-card-metric">88% + 99.75% · Classification</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="model-card">
            <div class="model-card-icon">💬</div>
            <div class="model-card-title">NLP Studio — BERT + GPT-2</div>
            <div class="model-card-desc">
                Dual-architecture NLP: BERT for sentiment classification on SST-2 
                (≥ 93% accuracy) and GPT-2 for controllable, domain-adaptive text generation.
            </div>
            <div class="model-card-metric">≥93% · NLP</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Architecture Overview ──
    st.markdown('<div class="section-header">Architecture Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Technical specifications across all models</div>', unsafe_allow_html=True)

    arch_data = pd.DataFrame({
        "Model": ["BostonHousingNet", "NEXUS-RT v1", "BERT-base", "GPT-2"],
        "Type": ["Residual MLP", "Dual-Head Residual MLP", "Transformer Encoder", "Transformer Decoder"],
        "Parameters": ["~20K", "174,515", "110M", "124M"],
        "Task": ["Regression", "Multi-Task Classification", "Sentiment Classification", "Text Generation"],
        "Framework": ["PyTorch", "PyTorch", "HuggingFace", "HuggingFace"],
        "Key Metric": ["R² > 0.80", "88% / 99.75%", "≥ 93% Acc", "Perplexity"],
    })

    st.dataframe(
        arch_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Parameters": st.column_config.TextColumn("Params", width="small"),
        }
    )

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        Krisha Patel — AI Model Hub v1.0 &nbsp;·&nbsp; Built with Streamlit, PyTorch & HuggingFace Transformers
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: T1 — HOUSING PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏘️  Housing Predictor":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">🏘️</div>
        <div class="page-header-text">Boston Housing Predictor</div>
    </div>
    <div class="page-desc">
        Predict median home values using a lightweight residual neural network trained 
        on 13 neighborhood features with engineered interactions.
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model ──
    try:
        from utils.t1_inference import load_t1_model, predict_price, get_feature_info

        with st.spinner("Loading BostonHousingNet..."):
            model_t1, ckpt_t1 = load_t1_model(BASE_DIR)

        feature_info = get_feature_info()

        # ── Model Info Expander ──
        with st.expander("📋  Model Details", expanded=False):
            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.metric("Architecture", "Residual MLP")
            ci2.metric("Parameters", f"{sum(p.numel() for p in model_t1.parameters()):,}")
            ci3.metric("Best R²", f"{ckpt_t1.get('best_r2', 0):.4f}")
            ci4.metric("Hidden Dim", "128")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Input Form ──
        st.markdown('<div class="section-header">Input Features</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subheader">Adjust the neighborhood characteristics to predict housing price</div>', unsafe_allow_html=True)

        feature_values = {}

        # Row 1: CRIM, ZN, INDUS, CHAS
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            feature_values["CRIM"] = st.number_input(
                "🔒 Crime Rate", min_value=0.0, max_value=100.0,
                value=feature_info["CRIM"]["default"], step=0.1,
                help="Per capita crime rate by town"
            )
        with r1c2:
            feature_values["ZN"] = st.number_input(
                "🏗️ Residential Zoning", min_value=0.0, max_value=100.0,
                value=feature_info["ZN"]["default"], step=1.0,
                help="Proportion of residential land zoned for lots over 25,000 sq.ft."
            )
        with r1c3:
            feature_values["INDUS"] = st.number_input(
                "🏭 Industrial Area", min_value=0.0, max_value=30.0,
                value=feature_info["INDUS"]["default"], step=0.5,
                help="Proportion of non-retail business acres per town"
            )
        with r1c4:
            feature_values["CHAS"] = st.selectbox(
                "🌊 Charles River", options=[0, 1],
                index=0, help="1 if tract bounds river; 0 otherwise"
            )

        # Row 2: NOX, RM, AGE, DIS
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            feature_values["NOX"] = st.number_input(
                "💨 NOX Concentration", min_value=0.3, max_value=0.9,
                value=feature_info["NOX"]["default"], step=0.01, format="%.3f",
                help="Nitric oxides concentration (parts per 10 million)"
            )
        with r2c2:
            feature_values["RM"] = st.number_input(
                "🛏️ Avg Rooms", min_value=3.0, max_value=9.0,
                value=feature_info["RM"]["default"], step=0.1, format="%.1f",
                help="Average number of rooms per dwelling"
            )
        with r2c3:
            feature_values["AGE"] = st.number_input(
                "📅 Building Age", min_value=0.0, max_value=100.0,
                value=feature_info["AGE"]["default"], step=1.0,
                help="Proportion of owner-occupied units built prior to 1940"
            )
        with r2c4:
            feature_values["DIS"] = st.number_input(
                "📍 Employment Distance", min_value=1.0, max_value=13.0,
                value=feature_info["DIS"]["default"], step=0.1, format="%.1f",
                help="Weighted distances to five Boston employment centres"
            )

        # Row 3: RAD, TAX, PTRATIO, B
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        with r3c1:
            feature_values["RAD"] = st.number_input(
                "🛣️ Highway Access", min_value=1, max_value=24,
                value=int(feature_info["RAD"]["default"]), step=1,
                help="Index of accessibility to radial highways"
            )
        with r3c2:
            feature_values["TAX"] = st.number_input(
                "💰 Tax Rate", min_value=180, max_value=720,
                value=int(feature_info["TAX"]["default"]), step=10,
                help="Full-value property-tax rate per $10,000"
            )
        with r3c3:
            feature_values["PTRATIO"] = st.number_input(
                "🎓 Pupil-Teacher Ratio", min_value=12.0, max_value=22.0,
                value=feature_info["PTRATIO"]["default"], step=0.1, format="%.1f",
                help="Pupil-teacher ratio by town"
            )
        with r3c4:
            feature_values["B"] = st.number_input(
                "👥 B Value", min_value=0.0, max_value=400.0,
                value=feature_info["B"]["default"], step=1.0,
                help="1000(Bk - 0.63)² where Bk is the proportion of Black residents"
            )

        # Row 4: LSTAT
        r4c1, r4c2, _, _ = st.columns(4)
        with r4c1:
            feature_values["LSTAT"] = st.number_input(
                "📊 Lower Status %", min_value=1.0, max_value=40.0,
                value=feature_info["LSTAT"]["default"], step=0.5,
                help="Percentage lower status of the population"
            )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Predict Button ──
        if st.button("🏠  Predict Housing Price", use_container_width=True, key="t1_predict"):
            with st.spinner("Running inference..."):
                predicted_price = predict_price(model_t1, ckpt_t1, feature_values)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Results ──
            rc1, rc2, rc3 = st.columns([2, 1, 1])

            with rc1:
                st.markdown(f"""
                <div class="result-card pulse-border">
                    <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; 
                         letter-spacing: 1.5px; font-weight: 600; margin-bottom: 8px;">
                        Predicted Median Value
                    </div>
                    <div class="result-value">${predicted_price:,.0f}</div>
                    <div class="result-label">in thousands (1970s dollars)</div>
                </div>
                """, unsafe_allow_html=True)

            with rc2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value" style="font-size: 1.3rem;">R² > 0.80</div>
                    <div class="stat-label">Model Accuracy</div>
                </div>
                """, unsafe_allow_html=True)

            with rc3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">⚡</div>
                    <div class="stat-value" style="font-size: 1.3rem;">~20K</div>
                    <div class="stat-label">Parameters</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Gauge Chart ──
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                number={"prefix": "$", "suffix": "K", "font": {"size": 48, "color": "#C4B5FD"}},
                gauge={
                    "axis": {"range": [0, 55], "tickwidth": 1, "tickcolor": "#4A5568",
                             "tickfont": {"color": "#94A3B8"}},
                    "bar": {"color": "#7C3AED", "thickness": 0.3},
                    "bgcolor": "rgba(26,26,46,0.3)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 15], "color": "rgba(239,68,68,0.15)"},
                        {"range": [15, 30], "color": "rgba(234,179,8,0.12)"},
                        {"range": [30, 55], "color": "rgba(34,197,94,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#A78BFA", "width": 3},
                        "thickness": 0.8,
                        "value": predicted_price,
                    }
                },
                title={"text": "Price Range Position", "font": {"size": 14, "color": "#94A3B8"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E2E8F0"},
                height=300,
                margin=dict(l=30, r=30, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Feature Contribution Bar ──
            feature_vals_list = [(k, v) for k, v in feature_values.items()]
            fig_bar = go.Figure(go.Bar(
                x=[v for _, v in feature_vals_list],
                y=[k for k, _ in feature_vals_list],
                orientation="h",
                marker=dict(
                    color=[v for _, v in feature_vals_list],
                    colorscale=[[0, "#312E81"], [0.5, "#7C3AED"], [1, "#C4B5FD"]],
                    line=dict(width=0),
                    cornerradius=4,
                ),
            ))
            fig_bar.update_layout(
                title={"text": "Input Feature Values", "font": {"size": 14, "color": "#94A3B8"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E2E8F0"},
                xaxis={"gridcolor": "rgba(124,58,237,0.08)", "title": "Value"},
                yaxis={"gridcolor": "rgba(124,58,237,0.08)"},
                height=400,
                margin=dict(l=10, r=10, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error loading T1 model: {str(e)}")
        st.info("Make sure the model checkpoint exists at `T1/saved_model/boston_housing_model.pth`")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: T2 — SALES ANALYTICS (NEXUS-RT)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊  Sales Analytics":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">📊</div>
        <div class="page-header-text">NEXUS-RT v1 — Sales Analytics</div>
    </div>
    <div class="page-desc">
        Dual-head neural network for simultaneous profitability prediction and product 
        sub-category classification on retail transaction data.
    </div>
    """, unsafe_allow_html=True)

    try:
        from utils.t2_inference import (
            load_t2_model, predict_transaction,
            get_ship_modes, get_segments, get_categories,
            get_subcategories, get_regions, get_states,
        )

        with st.spinner("Loading NEXUS-RT v1 and preparing data pipeline..."):
            model_t2, scaler_t2, le_t2, subcat_names_t2, feature_cols_t2 = load_t2_model(BASE_DIR)

        # ── Model Info ──
        with st.expander("📋  Model Details", expanded=False):
            mi1, mi2, mi3, mi4 = st.columns(4)
            mi1.metric("Architecture", "Dual-Head MLP")
            mi2.metric("Parameters", "174,515")
            mi3.metric("Binary Accuracy", "88.04%")
            mi4.metric("Multi-Class Acc", "99.75%")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subheader">Enter the retail transaction details for profitability and category prediction</div>', unsafe_allow_html=True)

        # ── Input Form ──
        t2r1c1, t2r1c2, t2r1c3, t2r1c4 = st.columns(4)
        with t2r1c1:
            ship_mode = st.selectbox("🚚 Ship Mode", get_ship_modes())
        with t2r1c2:
            segment = st.selectbox("👤 Segment", get_segments())
        with t2r1c3:
            category = st.selectbox("📁 Category", get_categories())
        with t2r1c4:
            sub_category = st.selectbox("📦 Sub-Category", get_subcategories())

        t2r2c1, t2r2c2, t2r2c3 = st.columns(3)
        with t2r2c1:
            region = st.selectbox("🌎 Region", get_regions())
        with t2r2c2:
            state = st.selectbox("📍 State", get_states())
        with t2r2c3:
            city = st.text_input("🏙️ City", value="New York")

        t2r3c1, t2r3c2, t2r3c3 = st.columns(3)
        with t2r3c1:
            sales = st.number_input("💲 Sales ($)", min_value=0.0, max_value=50000.0,
                                    value=250.0, step=10.0)
        with t2r3c2:
            quantity = st.number_input("📦 Quantity", min_value=1, max_value=14,
                                       value=3, step=1)
        with t2r3c3:
            discount = st.slider("🏷️ Discount", min_value=0.0, max_value=0.8,
                                  value=0.1, step=0.05)

        t2r4c1, t2r4c2 = st.columns(2)
        with t2r4c1:
            order_date = st.date_input("📅 Order Date", value=datetime.now().date())
        with t2r4c2:
            ship_date = st.date_input("📅 Ship Date",
                                       value=(datetime.now() + timedelta(days=5)).date())

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        if st.button("📊  Analyze Transaction", use_container_width=True, key="t2_predict"):
            raw_input = {
                "Ship Mode": ship_mode,
                "Segment": segment,
                "Category": category,
                "Sub-Category": sub_category,
                "Region": region,
                "State": state,
                "City": city,
                "Sales": sales,
                "Quantity": quantity,
                "Discount": discount,
                "Order Date": str(order_date),
                "Ship Date": str(ship_date),
                "Row ID": 1,
                "Order ID": "PRED-001",
                "Customer ID": "PRED-CUST",
                "Customer Name": "Prediction User",
                "Country": "United States",
                "Postal Code": 10001,
                "Product ID": "PRED-PROD",
                "Product Name": "Predicted Product",
                "Profit": 0,
            }

            with st.spinner("Running dual-head inference..."):
                result = predict_transaction(
                    model_t2, scaler_t2, le_t2, feature_cols_t2, subcat_names_t2, raw_input
                )

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Results Row ──
            res1, res2 = st.columns(2)

            with res1:
                is_prof = result["is_profitable"]
                conf = result["profit_confidence"]
                cls = "profit-yes" if is_prof else "profit-no"
                label = "✅ Profitable" if is_prof else "❌ Not Profitable"
                color = "#22C55E" if is_prof else "#EF4444"
                st.markdown(f"""
                <div class="{cls}">
                    <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;
                         letter-spacing: 1.5px; font-weight: 600; margin-bottom: 8px;">
                        Head A — Profitability
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: {color}; margin-bottom: 4px;">
                        {label}
                    </div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">
                        Confidence: {conf*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res2:
                subcat = result["sub_category"]
                subcat_conf = result["subcat_confidence"]
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;
                         letter-spacing: 1.5px; font-weight: 600; margin-bottom: 8px;">
                        Head B — Sub-Category
                    </div>
                    <div class="result-value" style="font-size: 2rem;">{subcat}</div>
                    <div class="result-label">Confidence: {subcat_conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Probability Charts ──
            p_col1, p_col2 = st.columns(2)

            with p_col1:
                probs = result["profit_probs"]
                fig_prof = go.Figure(go.Bar(
                    x=["Not Profitable", "Profitable"],
                    y=probs,
                    marker=dict(
                        color=["#EF4444", "#22C55E"],
                        line=dict(width=0),
                        cornerradius=6,
                    ),
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition="outside",
                    textfont=dict(color="#E2E8F0"),
                ))
                fig_prof.update_layout(
                    title={"text": "Profitability Probabilities", "font": {"size": 14, "color": "#94A3B8"}},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#E2E8F0"},
                    yaxis={"range": [0, 1.15], "gridcolor": "rgba(124,58,237,0.08)", "title": "Probability"},
                    xaxis={"gridcolor": "rgba(124,58,237,0.08)"},
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=20),
                )
                st.plotly_chart(fig_prof, use_container_width=True)

            with p_col2:
                subcat_probs = result["subcat_probs"]
                top_n = dict(sorted(subcat_probs.items(), key=lambda x: x[1], reverse=True)[:8])
                fig_subcat = go.Figure(go.Bar(
                    y=list(top_n.keys()),
                    x=list(top_n.values()),
                    orientation="h",
                    marker=dict(
                        color=list(top_n.values()),
                        colorscale=[[0, "#312E81"], [0.5, "#7C3AED"], [1, "#C4B5FD"]],
                        line=dict(width=0),
                        cornerradius=4,
                    ),
                    text=[f"{p*100:.1f}%" for p in top_n.values()],
                    textposition="outside",
                    textfont=dict(color="#E2E8F0"),
                ))
                fig_subcat.update_layout(
                    title={"text": "Top Sub-Category Probabilities", "font": {"size": 14, "color": "#94A3B8"}},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#E2E8F0"},
                    xaxis={"range": [0, 1.15], "gridcolor": "rgba(124,58,237,0.08)", "title": "Probability"},
                    yaxis={"gridcolor": "rgba(124,58,237,0.08)"},
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=20),
                )
                st.plotly_chart(fig_subcat, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error loading T2 model: {str(e)}")
        st.info("Make sure `T2/nexus_rt_v1.pth` and `T2/Dataset/Sample - Superstore.csv` exist.")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: T3 — NLP STUDIO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💬  NLP Studio":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">💬</div>
        <div class="page-header-text">NLP Studio</div>
    </div>
    <div class="page-desc">
        Dual-architecture NLP: BERT for sentiment analysis and GPT-2 for 
        controllable text generation. Explore both the encoder and decoder paradigms.
    </div>
    """, unsafe_allow_html=True)

    # Note about first-time downloads
    st.markdown("""
    <div style="padding: 12px 16px; background: rgba(124,58,237,0.06); 
         border-radius: 10px; border: 1px solid rgba(124,58,237,0.12);
         margin-bottom: 24px;">
        <span style="color: #A78BFA; font-weight: 600;">ℹ️ Note:</span>
        <span style="color: #94A3B8; font-size: 0.88rem;">
            Models are loaded on first use. The initial download may take a few minutes 
            (BERT ~440 MB, GPT-2 ~500 MB). Subsequent uses are instant (cached).
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab_bert, tab_gpt2 = st.tabs(["🎯  BERT — Sentiment Analysis", "✍️  GPT-2 — Text Generation"])

    # ── BERT Tab ──
    with tab_bert:
        try:
            from utils.t3_inference import SAMPLE_SENTIMENTS

            with st.expander("📋  Model Details", expanded=False):
                bi1, bi2, bi3 = st.columns(3)
                bi1.metric("Model", "BERT-base-uncased")
                bi2.metric("Parameters", "110M")
                bi3.metric("Fine-tuned on", "SST-2")

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Sentiment Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subheader">Enter text to classify its sentiment as positive or negative</div>', unsafe_allow_html=True)

            # Sample sentences selector
            sample_choice = st.selectbox(
                "💡 Try a sample sentence (or type your own below)",
                ["— Custom Input —"] + SAMPLE_SENTIMENTS,
                key="bert_sample"
            )

            if sample_choice != "— Custom Input —":
                input_text = st.text_area("📝 Input Text", value=sample_choice, height=100, key="bert_input")
            else:
                input_text = st.text_area("📝 Input Text",
                    value="This movie was absolutely fantastic and I loved every moment of it!",
                    height=100, key="bert_input_custom")

            if st.button("🎯  Classify Sentiment", use_container_width=True, key="bert_predict"):
                if input_text.strip():
                    # Lazy-load BERT only when button is clicked
                    from utils.t3_inference import load_bert_model, predict_sentiment

                    with st.spinner("Loading BERT model (first time may take a few minutes)..."):
                        bert_model, bert_tokenizer, bert_device = load_bert_model()

                    with st.spinner("Analyzing sentiment..."):
                        result = predict_sentiment(bert_model, bert_tokenizer, bert_device, input_text)

                    if result.get("error"):
                        st.error(f"Inference error: {result['error']}")
                    else:
                        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

                        sc1, sc2 = st.columns([1, 1])

                        with sc1:
                            is_pos = result["label"] == "Positive"
                            cls = "sentiment-positive" if is_pos else "sentiment-negative"
                            lbl_cls = "sentiment-label-pos" if is_pos else "sentiment-label-neg"
                            emoji = "😊" if is_pos else "😞"
                            st.markdown(f"""
                            <div class="{cls}">
                                <div style="font-size: 3rem; margin-bottom: 8px;">{emoji}</div>
                                <div class="{lbl_cls}">{result["label"]}</div>
                                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 8px;">
                                    Confidence: {result["confidence"]*100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with sc2:
                            fig_sent = go.Figure(go.Bar(
                                x=["Negative", "Positive"],
                                y=[result["negative_prob"], result["positive_prob"]],
                                marker=dict(
                                    color=["#EF4444", "#22C55E"],
                                    line=dict(width=0),
                                    cornerradius=8,
                                ),
                                text=[f"{result['negative_prob']*100:.1f}%", f"{result['positive_prob']*100:.1f}%"],
                                textposition="outside",
                                textfont=dict(color="#E2E8F0", size=14),
                            ))
                            fig_sent.update_layout(
                                title={"text": "Class Probabilities", "font": {"size": 14, "color": "#94A3B8"}},
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font={"color": "#E2E8F0"},
                                yaxis={"range": [0, 1.15], "gridcolor": "rgba(124,58,237,0.08)", "title": "Probability"},
                                xaxis={"gridcolor": "rgba(124,58,237,0.08)"},
                                height=300,
                                margin=dict(l=10, r=10, t=50, b=20),
                            )
                            st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.warning("Please enter some text to analyze.")

        except Exception as e:
            st.error(f"⚠️ Error with BERT: {str(e)}")
            st.info("The BERT model requires the `transformers` package. Run: `pip install transformers`")

    # ── GPT-2 Tab ──
    with tab_gpt2:
        try:
            from utils.t3_inference import SAMPLE_PROMPTS

            with st.expander("📋  Model Details", expanded=False):
                gi1, gi2, gi3 = st.columns(3)
                gi1.metric("Model", "GPT-2")
                gi2.metric("Parameters", "124M")
                gi3.metric("Type", "Autoregressive")

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Text Generation</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subheader">Provide a prompt and adjust generation parameters</div>', unsafe_allow_html=True)

            # Sample prompts
            domain_choice = st.selectbox(
                "💡 Choose a domain prompt (or write your own below)",
                ["— Custom Prompt —"] + list(SAMPLE_PROMPTS.keys()),
                key="gpt2_domain"
            )

            if domain_choice != "— Custom Prompt —":
                prompt_text = st.text_area("✍️ Prompt", value=SAMPLE_PROMPTS[domain_choice],
                                            height=100, key="gpt2_prompt_sample")
            else:
                prompt_text = st.text_area("✍️ Prompt",
                    value="The future of artificial intelligence lies in",
                    height=100, key="gpt2_prompt_custom")

            # Generation parameters
            with st.expander("⚙️  Generation Parameters", expanded=True):
                gp1, gp2, gp3, gp4 = st.columns(4)
                with gp1:
                    max_tokens = st.slider("Max Tokens", 20, 250, 100, step=10)
                with gp2:
                    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1,
                        help="Low = deterministic, High = creative")
                with gp3:
                    top_p = st.slider("Top-p", 0.5, 1.0, 0.92, step=0.02,
                        help="Nucleus sampling threshold")
                with gp4:
                    top_k = st.slider("Top-k", 10, 100, 50, step=5,
                        help="Limits token selection to top-k most likely")

            if st.button("✍️  Generate Text", use_container_width=True, key="gpt2_generate"):
                if prompt_text.strip():
                    # Lazy-load GPT-2 only when button is clicked
                    from utils.t3_inference import load_gpt2_model, generate_text

                    with st.spinner("Loading GPT-2 model (first time may take a few minutes)..."):
                        gpt2_model, gpt2_tokenizer, gpt2_device = load_gpt2_model()

                    with st.spinner("Generating text..."):
                        result = generate_text(
                            gpt2_model, gpt2_tokenizer, gpt2_device,
                            prompt_text, max_tokens, temperature, top_p, top_k,
                        )

                    if result.get("error"):
                        st.error(f"Generation error: {result['error']}")
                    else:
                        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

                        # Stats row
                        gs1, gs2, gs3 = st.columns(3)
                        with gs1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">⏱️</div>
                                <div class="stat-value" style="font-size:1.3rem;">{result['time']:.2f}s</div>
                                <div class="stat-label">Generation Time</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with gs2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">📊</div>
                                <div class="stat-value" style="font-size:1.3rem;">{result['num_tokens']}</div>
                                <div class="stat-label">Output Tokens</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with gs3:
                            tokens_per_sec = result['num_tokens'] / max(result['time'], 0.01)
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">⚡</div>
                                <div class="stat-value" style="font-size:1.3rem;">{tokens_per_sec:.0f}</div>
                                <div class="stat-label">Tokens / Second</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("", unsafe_allow_html=True)

                        # Generated text display
                        st.markdown(f"""
                        <div class="gen-box">
                            <div style="font-size: 0.75rem; color: #7C3AED; text-transform: uppercase; 
                                 letter-spacing: 1.5px; font-weight: 600; margin-bottom: 12px;">
                                Generated Output
                            </div>
                            {result['text']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a prompt.")

        except Exception as e:
            st.error(f"⚠️ Error with GPT-2: {str(e)}")
            st.info("The GPT-2 model requires the `transformers` package. Run: `pip install transformers`")

