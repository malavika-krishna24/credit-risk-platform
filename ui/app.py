"""
ui/app.py

Main Streamlit application for the Credit Risk Intelligence Platform.
Five sections as required: EDA, Risk Prediction, Explainability, Business Rules,
Talk-to-Data chatbot. Charts are interactive (Plotly) rather than static images
wherever the underlying data supports it.
"""
import sys
sys.path.insert(0, ".")

import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.ml.predict import predict_risk, prepare_features
from src.ml.explain import explain_prediction, plain_english_summary
from src.utils.config import MODELS_DIR, risk_band
from src.utils.helpers import format_percent

st.set_page_config(page_title="Credit Risk Intelligence Platform", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------------
# Design system — deliberate dark navy/teal palette, IBM Plex Sans throughout.
# Includes entrance animations and hover interactions (fadeInUp on cards,
# lift-on-hover, animated risk badges) so the interface feels alive rather
# than a static report.
# ---------------------------------------------------------------------------
GREEN, AMBER, RED, TEAL = "#4C9A6B", "#E0A458", "#C0392B", "#3FA796"
BG, CARD, TEXT, MUTED = "#0B1420", "#131F30", "#E8EDF3", "#8A97AB"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { max-width: 1150px; padding-top: 2rem; }
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; letter-spacing: -0.01em; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

.risk-card, .rule-card, .factor-card, .kpi-card {
    padding: 1.5rem 1.75rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    background: #131F30;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease-out both;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.risk-card:hover, .rule-card:hover, .factor-card:hover, .kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(63,167,150,0.18);
    border-color: #3FA796;
}

.risk-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    animation: fadeInUp 0.4s ease-out both;
}
.risk-low { background: rgba(76,154,107,0.18); color: #6FCB9A; border: 1px solid #4C9A6B; }
.risk-medium { background: rgba(224,164,88,0.18); color: #F0C079; border: 1px solid #E0A458; }
.risk-high { background: rgba(192,57,43,0.18); color: #F08A7E; border: 1px solid #C0392B; }

.live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #3FA796; margin-right: 6px; animation: pulse 2s ease-in-out infinite;
}

.hero {
    padding: 2rem 2.3rem;
    border-radius: 14px;
    background: linear-gradient(120deg, #0B1420 0%, #132436 45%, #0F3A34 100%);
    background-size: 200% 200%;
    animation: gradientShift 10s ease infinite, fadeInUp 0.6s ease-out both;
    border: 1px solid rgba(63,167,150,0.25);
    margin-bottom: 1.4rem;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero h1 { color: #F2F6FA; }
.hero p { color: #A9B7C6; font-size: 1.02rem; max-width: 760px; }

.kpi-number { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 500; color: #E8EDF3; }
.kpi-label { font-size: 0.85rem; color: #8A97AB; margin-bottom: 0.2rem; }
.insight-line { border-left: 3px solid #3FA796; padding-left: 0.9rem; margin: 0.6rem 0; color: #C7D1DD; animation: fadeInUp 0.5s ease-out both; }

.chat-bubble-user {
    background: #1B2A40; border: 1px solid #2A3A50; border-radius: 12px 12px 2px 12px;
    padding: 0.8rem 1.1rem; margin: 0.6rem 0; animation: fadeInUp 0.35s ease-out both; color: #E8EDF3;
}
.chat-bubble-answer {
    background: rgba(63,167,150,0.10); border: 1px solid rgba(63,167,150,0.35); border-radius: 12px 12px 12px 2px;
    padding: 0.9rem 1.1rem; margin: 0.6rem 0 1.2rem 0; animation: fadeInUp 0.45s ease-out 0.1s both; color: #E8EDF3;
}

/* Streamlit's own buttons/inputs pick up a subtle lift too */
.stButton > button { transition: transform 0.15s ease, box-shadow 0.15s ease; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(63,167,150,0.25); }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": TEXT, "family": "IBM Plex Sans"},
    margin=dict(l=10, r=20, t=50, b=10),
)


def count_up_html(target: float, label: str, decimals: int = 0, suffix: str = "", duration_ms: int = 900) -> str:
    """Self-contained JS component: animates 0 -> target on render.

    IMPORTANT: components.html renders inside its own sandboxed <iframe>, which
    does NOT inherit the page-level <style> block injected via st.markdown.
    Every rule the card needs (colors, font, background, animation) must be
    declared right here, or text renders in the browser's default black on a
    transparent background and disappears against the dark theme.
    """
    elem_id = f"cu_{abs(hash((label, target)))}"
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
        html, body {{ margin: 0; padding: 0; background: transparent; }}
        .cu-card {{
            font-family: 'IBM Plex Sans', sans-serif;
            padding: 1.2rem 1.4rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            background: #131F30;
            box-sizing: border-box;
            opacity: 0;
            transform: translateY(10px) scale(0.98);
            animation: cuFadeIn 0.5s ease forwards;
        }}
        @keyframes cuFadeIn {{
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        .cu-label {{ font-size: 0.85rem; color: #8A97AB; margin-bottom: 0.3rem; }}
        .cu-number {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 2.1rem; font-weight: 500; color: #E8EDF3;
        }}
    </style>
    <div class="cu-card">
        <div class="cu-label">{label}</div>
        <div class="cu-number" id="{elem_id}">0{suffix}</div>
    </div>
    <script>
    (function() {{
        const el = document.getElementById("{elem_id}");
        const target = {target};
        const duration = {duration_ms};
        const decimals = {decimals};
        const start = performance.now();
        function step(ts) {{
            const progress = Math.min((ts - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const val = target * eased;
            const formatted = val.toLocaleString('en-US', {{ minimumFractionDigits: decimals, maximumFractionDigits: decimals }});
            el.textContent = formatted + "{suffix}";
            if (progress < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }})();
    </script>
    """


@st.cache_data
def load_features():
    return pd.read_parquet("data/features_train.parquet")


@st.cache_resource
def load_model_and_meta():
    model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    return model, meta


def risk_badge_html(band: str) -> str:
    cls = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[band]
    return f'<span class="risk-badge {cls}">{band} Risk</span>'


def risk_gauge(proba: float, band: str):
    """Interactive Plotly gauge — colored zones matching the risk bands, with
    an animated needle-style threshold marker at the actual score."""
    color = {"Low": GREEN, "Medium": AMBER, "High": RED}[band]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 42, "color": TEXT, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 11}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 10], "color": "rgba(76,154,107,0.20)"},
                {"range": [10, 35], "color": "rgba(224,164,88,0.20)"},
                {"range": [35, 100], "color": "rgba(192,57,43,0.20)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.9, "value": proba * 100},
        },
    ))
    fig.update_layout(**{**PLOTLY_LAYOUT, "height": 260, "margin": dict(l=25, r=25, t=35, b=10)})
    return fig


def shap_diverging_bar(explanation: dict):
    """Interactive horizontal diverging bar chart of SHAP contributions —
    replaces the old static list, adds hover tooltips and exact values."""
    feats = [f["feature"] for f in explanation["top_features"]][::-1]
    vals = [f["shap_contribution"] for f in explanation["top_features"]][::-1]
    raw_vals = [f["value"] for f in explanation["top_features"]][::-1]
    colors = [RED if v > 0 else GREEN for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        customdata=raw_vals,
        hovertemplate="<b>%{y}</b><br>value = %{customdata}<br>SHAP impact = %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "height": 340, "margin": dict(l=10, r=60, t=20, b=30)},
        xaxis_title="SHAP contribution (negative = lowers risk, positive = raises risk)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.25)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
    )
    return fig


def styled_bar(x, y, title, orientation="v", colors=None, xlabel=None, ylabel=None, height=380):
    if orientation == "v":
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors or TEAL,
                                text=[f"{v:.1f}" for v in y], textposition="outside"))
    else:
        fig = go.Figure(go.Bar(x=x, y=y, orientation="h", marker_color=colors or TEAL,
                                text=[f"{v:.1f}" for v in x], textposition="outside"))
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "height": height},
        title={"text": title, "font": {"size": 15, "color": TEXT}},
        xaxis=dict(title=xlabel, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=ylabel, gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ◆ Credit Risk Platform")
st.sidebar.caption("AI-powered credit risk intelligence — NeoStats assignment")
page = st.sidebar.radio(
    "Section",
    ["🏠 Overview", "📊 Data Exploration (EDA)", "🎯 Risk Prediction", "🔍 Explainability", "📋 Business Rules", "💬 Talk to Data"],
    label_visibility="collapsed",
)

df = load_features()
model, meta = load_model_and_meta()

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.markdown(
        '<div class="hero"><h1 style="margin:0;">Credit Risk Intelligence Platform</h1>'
        '<p style="margin-top:0.6rem;">A lightweight, explainable platform for scoring loan-default risk — '
        'multi-table feature engineering, a calibrated LightGBM model, SHAP explainability, derived '
        'business rules, and a guardrailed natural-language chatbot over the underlying data.</p></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    kpis = [
        (len(df), "Applicants", 0, ""),
        (df['TARGET'].mean() * 100, "Default Rate", 1, "%"),
        (meta['metrics']['roc_auc'], "Model ROC-AUC", 3, ""),
        (len(meta['feature_names']), "Features Used", 0, ""),
    ]
    for col, (value, label, decimals, suffix) in zip(cols, kpis):
        with col:
            components.html(count_up_html(value, label, decimals=decimals, suffix=suffix), height=118)

    st.markdown("### How this platform is put together")
    st.markdown("""
    - **Data Exploration** — dataset summary, data quality, and business insights from the raw + joined tables
    - **Risk Prediction** — score a new applicant and get a Low / Medium / High risk band on a live gauge
    - **Explainability** — SHAP-based, plain-English reasons behind any prediction, ranked visually
    - **Business Rules** — auditable IF-THEN rules mined from the model, with measured lift
    - **Talk to Data** — ask questions about the applicant data in plain English, chat-style
    """)

# ---------------------------------------------------------------------------
# PAGE: EDA — all charts are interactive Plotly, computed live from the data
# ---------------------------------------------------------------------------
elif page == "📊 Data Exploration (EDA)":
    st.title("Data Exploration & Insights")
    st.caption("Home Credit application data joined with bureau, previous-application, POS/cash, and credit-card history. Hover any chart to explore the exact numbers.")

    # 1. Target distribution
    counts = df["TARGET"].value_counts()
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = styled_bar(["Repaid (0)", "Default (1)"], counts.values,
                          "Target Class Distribution — Severe Imbalance (~8% default)",
                          colors=[GREEN, RED], ylabel="Applicant Count", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="insight-line">Only <b>{df["TARGET"].mean()*100:.1f}%</b> of applicants '
                     f'default — this drives the ROC-AUC/PR-AUC evaluation choice and imbalance handling '
                     f'in the ML pipeline.</div>', unsafe_allow_html=True)

    # 2. Default rate by age group
    d = df.copy()
    d["age_group"] = pd.cut(d["YEARS_BIRTH"], bins=[20, 30, 40, 50, 60, 70],
                             labels=["20-30", "30-40", "40-50", "50-60", "60-70"])
    grp = (d.groupby("age_group", observed=True)["TARGET"].mean() * 100)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = styled_bar(grp.index.astype(str), grp.values, "Default Rate by Age Group", ylabel="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="insight-line">Default rate falls steadily with age, from '
                     f'<b>{grp.iloc[0]:.1f}%</b> (20-30) to <b>{grp.iloc[-1]:.1f}%</b> (60-70).</div>',
                     unsafe_allow_html=True)

    # 3. Default rate by income type
    grp = df.groupby("NAME_INCOME_TYPE")["TARGET"].agg(["mean", "count"])
    grp = grp[grp["count"] > 50].sort_values("mean", ascending=True) * [100, 1]
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = styled_bar(grp["mean"].values, grp.index.astype(str), "Default Rate by Income Type",
                          orientation="h", xlabel="Default Rate (%)", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top = grp.sort_values("mean", ascending=False).iloc[0]
        st.markdown(f'<div class="insight-line"><b>{top.name}</b> applicants default at '
                     f'<b>{top["mean"]:.1f}%</b> — dramatically higher than salaried/pensioner applicants.</div>',
                     unsafe_allow_html=True)

    # 4. Bureau overdue history vs default
    d = df.copy()
    d["bureau_overdue"] = np.where(d["bureau_overdue_loan_count"].fillna(0) > 0, "Has Overdue Bureau Loan", "No Overdue Bureau Loan")
    grp = d.groupby("bureau_overdue")["TARGET"].mean() * 100
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = styled_bar(grp.index, grp.values, "External Bureau History vs Default Rate",
                          colors=[RED, GREEN], ylabel="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        lift = grp.max() / grp.min()
        st.markdown(f'<div class="insight-line">An overdue bureau loan is associated with a '
                     f'<b>{lift:.1f}x</b> higher default rate — one of the strongest signals in the dataset.</div>',
                     unsafe_allow_html=True)

    # 5. Prior refusal vs default
    d = df.copy()
    d["refused"] = np.where(d["prev_refused_count"].fillna(0) > 0, "Previously Refused", "Never Refused")
    grp = d.groupby("refused")["TARGET"].mean() * 100
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = styled_bar(grp.index, grp.values, "Prior Loan Refusal vs Default Rate",
                          colors=[RED, GREEN], ylabel="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="insight-line">A single past refusal with this lender predicts future '
                     f'default — even one rejection materially changes risk.</div>', unsafe_allow_html=True)

    # 6. Credit-income ratio distribution
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure()
        for target, color, label in [(0, GREEN, "Repaid"), (1, RED, "Default")]:
            subset = df[(df["TARGET"] == target) & (df["CREDIT_INCOME_RATIO"] < 15)]["CREDIT_INCOME_RATIO"]
            fig.add_trace(go.Histogram(x=subset, name=label, marker_color=color, opacity=0.55, histnorm="probability density", nbinsx=40))
        fig.update_layout(**{**PLOTLY_LAYOUT, "height": 380}, barmode="overlay",
                           title={"text": "Credit-to-Income Ratio Distribution by Outcome", "font": {"size": 15, "color": TEXT}},
                           xaxis=dict(title="Credit / Income Ratio", gridcolor="rgba(255,255,255,0.08)"),
                           yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.08)"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="insight-line">Defaulting applicants skew toward higher credit-to-income '
                     'ratios — borrowing more relative to what they earn.</div>', unsafe_allow_html=True)

    # 7. Missing data overview
    missing = (df.isnull().mean() * 100).sort_values(ascending=False).head(15)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = styled_bar(missing.values[::-1], missing.index[::-1], "Top 15 Columns by Missing Data %",
                          orientation="h", colors="#95A5A6", xlabel="% Missing", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="insight-line">~50 housing-quality columns are 50-70% missing — '
                     f'structural (tied to housing type), handled via the model\'s native missing-value '
                     f'support rather than imputation.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Risk Prediction
# ---------------------------------------------------------------------------
elif page == "🎯 Risk Prediction":
    st.title("Score a New Applicant")
    st.caption("Fill in the key fields below — the remaining model features are filled with population "
               "medians/modes, matching how the model treats sparsely-available data in production.")

    numeric_defaults = df.drop(columns=["TARGET"]).median(numeric_only=True)
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    categorical_defaults = df[categorical_cols].mode().iloc[0]
    defaults = pd.concat([numeric_defaults, categorical_defaults])

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input("Annual Income", min_value=0, value=150000, step=10000)
            credit = st.number_input("Loan Amount (Credit)", min_value=0, value=500000, step=10000)
            annuity = st.number_input("Annuity (Monthly Payment)", min_value=0, value=25000, step=1000)
            age_years = st.slider("Applicant Age", 18, 75, 35)
        with c2:
            income_type = st.selectbox("Income Type", sorted(df["NAME_INCOME_TYPE"].dropna().unique()))
            education = st.selectbox("Education", sorted(df["NAME_EDUCATION_TYPE"].dropna().unique()))
            employed_years = st.slider("Years Employed", 0, 45, 5)
            gender = st.selectbox("Gender", ["M", "F"])
        with c3:
            bureau_overdue = st.number_input("Overdue Bureau Loans", min_value=0, value=0, step=1)
            bureau_active = st.number_input("Active Bureau Loans", min_value=0, value=2, step=1)
            prev_refused = st.number_input("Prior Refusals (this lender)", min_value=0, value=0, step=1)
            ext_score = st.slider("Avg. External Credit Score (0-1)", 0.0, 1.0, 0.5)

        submitted = st.form_submit_button("Score Applicant", use_container_width=True)

    if submitted:
        applicant = defaults.to_dict()
        applicant.update({
            "SK_ID_CURR": 999999999,
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": -int(age_years * 365.25), "YEARS_BIRTH": age_years,
            "DAYS_EMPLOYED": -int(employed_years * 365.25), "YEARS_EMPLOYED": employed_years,
            "NAME_INCOME_TYPE": income_type, "NAME_EDUCATION_TYPE": education, "CODE_GENDER": gender,
            "bureau_overdue_loan_count": bureau_overdue, "bureau_active_loan_count": bureau_active,
            "prev_refused_count": prev_refused,
            "EXT_SOURCE_1": ext_score, "EXT_SOURCE_2": ext_score, "EXT_SOURCE_3": ext_score,
            "CREDIT_INCOME_RATIO": credit / max(income, 1), "ANNUITY_INCOME_RATIO": annuity / max(income, 1),
            "CREDIT_TERM_YEARS": credit / max(annuity, 1) / 12,
        })
        applicant_df = pd.DataFrame([applicant])
        result = predict_risk(applicant_df)
        proba = result.iloc[0]["default_probability"]
        band = result.iloc[0]["risk_band"]

        st.session_state["last_applicant_df"] = applicant_df
        st.session_state["last_prediction"] = (proba, band)

        st.markdown("---")
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.plotly_chart(risk_gauge(proba, band), use_container_width=True)
            st.markdown(f'<div style="text-align:center;">{risk_badge_html(band)}</div>', unsafe_allow_html=True)
        with col2:
            st.write("**What this means:**")
            if band == "Low":
                st.write("This applicant profile is in the Low risk band — default probability is below the "
                         "10% policy threshold. Standard approval workflow applies.")
            elif band == "Medium":
                st.write("This applicant profile is in the Medium risk band — consider additional verification "
                         "or adjusted terms (e.g. higher down payment, shorter term).")
            else:
                st.write("This applicant profile is in the High risk band — recommend manual underwriter review "
                         "before approval.")
            st.info("Go to the **Explainability** tab to see exactly which factors drove this score.")

# ---------------------------------------------------------------------------
# PAGE: Explainability
# ---------------------------------------------------------------------------
elif page == "🔍 Explainability":
    st.title("Explainable AI")
    st.caption("SHAP-based explanation of individual predictions — auditable, feature-level reasoning.")

    if "last_applicant_df" not in st.session_state:
        st.warning("Score an applicant on the **Risk Prediction** page first, then come back here to see why.")
    else:
        applicant_df = st.session_state["last_applicant_df"]
        proba, band = st.session_state["last_prediction"]

        X = prepare_features(applicant_df, meta)
        explanation = explain_prediction(model, X, top_n=8)

        st.markdown(f'{risk_badge_html(band)} &nbsp; **Default probability: {format_percent(proba)}**', unsafe_allow_html=True)
        st.write("")
        st.write("**Summary:** " + plain_english_summary(explanation))
        st.write("")
        st.write("**Feature contributions** (hover for exact values):")
        st.plotly_chart(shap_diverging_bar(explanation), use_container_width=True)

        st.markdown("### Global Feature Importance")
        st.caption("How each feature affects predictions across the whole applicant population.")
        st.image("documents/eval_charts/shap_summary.png", use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Business Rules
# ---------------------------------------------------------------------------
elif page == "📋 Business Rules":
    st.title("Derived Business Rules")
    st.caption("Plain-English underwriting rules mined from the model's learned patterns, each backed by "
               "measured lift on real validation data — not just model-internal importance.")

    try:
        rules_df = pd.read_csv("documents/derived_business_rules.csv")

        c1, c2 = st.columns([2, 1])
        with c1:
            top10 = rules_df.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top10["lift"], y=top10["condition"], orientation="h",
                marker_color=TEAL, text=[f"{v:.2f}x" for v in top10["lift"]], textposition="outside",
                hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}x<extra></extra>",
            ))
            fig.update_layout(**{**PLOTLY_LAYOUT, "height": 420},
                               title={"text": "Top 10 Rules by Risk Lift", "font": {"size": 15, "color": TEXT}},
                               xaxis=dict(title="Lift vs baseline", gridcolor="rgba(255,255,255,0.08)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.0)", automargin=True))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f'<div class="insight-line">Highest-lift rule: applicants matching '
                         f'<b>{rules_df.iloc[0]["condition"]}</b> default at '
                         f'<b>{rules_df.iloc[0]["lift"]:.2f}x</b> the baseline rate.</div>', unsafe_allow_html=True)

        st.markdown("### All Rules")
        for i, (_, r) in enumerate(rules_df.head(15).iterrows()):
            st.markdown(
                f'<div class="rule-card" style="padding:1rem 1.3rem; animation-delay:{min(i*0.04, 0.4)}s;">'
                f'If <b>{r["condition"]}</b><br>'
                f'&rarr; default rate <b>{r["default_rate"]*100:.1f}%</b> vs '
                f'{r["baseline_default_rate"]*100:.1f}% baseline '
                f'&nbsp;<span class="risk-badge risk-high">{r["lift"]}x lift</span> '
                f'<span style="color:#8A97AB;">(n={int(r["support"]):,})</span>'
                f'</div>', unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning("Run `python -m src.ml.rules` first to generate the business rules file.")

# ---------------------------------------------------------------------------
# PAGE: Talk to Data
# ---------------------------------------------------------------------------
elif page == "💬 Talk to Data":
    st.title("Talk to Your Data")
    st.caption("Ask questions in plain English — answered by an LLM-generated, validated SQL query "
               "against the real applicant database.")

    from src.talk_to_data.nl_to_sql import ask, SAMPLE_QUESTIONS
    from src.utils.config import GROQ_API_KEY, DB_PATH
    import time as _time

    if "demo" in DB_PATH.lower():
        st.info("This public demo queries a real, randomly-sampled subset of applicants "
                 "(~30,000 of 307,511) for deployment size reasons — every row is genuine data, "
                 "just not the full dataset. The model, EDA, and business rules elsewhere in this "
                 "app are trained/computed on the complete 307,511-applicant dataset. "
                 "See the README for the full local/Docker setup with the complete database.")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file (free key at console.groq.com) to use this feature.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []  # list of dicts: role, content, sql, data

        st.write("**Try one of these, or type your own question:**")
        cols = st.columns(3)
        sample_clicked = None
        for i, q in enumerate(SAMPLE_QUESTIONS):
            label = q if len(q) <= 60 else q[:57] + "..."
            if cols[i % 3].button(label, key=f"sample_{i}", use_container_width=True, help=q):
                sample_clicked = q

        # Replay prior turns as chat bubbles
        for turn in st.session_state["chat_history"]:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])
                if turn.get("sql"):
                    with st.expander("SQL query used"):
                        st.code(turn["sql"], language="sql")
                if turn.get("data") is not None and len(turn["data"]) > 0:
                    st.dataframe(turn["data"], use_container_width=True)

        question = sample_clicked or st.chat_input("Ask a question about the applicant data...")

        if question:
            with st.chat_message("user"):
                st.write(question)
            st.session_state["chat_history"].append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                steps = ["🧠 Understanding your question...", "🔎 Generating SQL query...",
                         "🛡️ Validating against safety rules...", "⚡ Querying the database..."]
                for step in steps[:2]:
                    status_placeholder.markdown(f'<div class="insight-line">{step}</div>', unsafe_allow_html=True)
                    _time.sleep(0.3)
                result = ask(question)
                status_placeholder.empty()

                if result["sql"]:
                    with st.expander("SQL query used"):
                        st.code(result["sql"], language="sql")

                # word-by-word reveal for a more conversational feel
                placeholder = st.empty()
                shown = ""
                for word in result["answer"].split(" "):
                    shown += word + " "
                    placeholder.markdown(shown)
                    _time.sleep(0.012)

                if result["success"] and result["data"] is not None and len(result["data"]) > 0:
                    st.dataframe(result["data"], use_container_width=True)

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": result["answer"],
                "sql": result["sql"],
                "data": result["data"] if result["success"] else None,
            })
