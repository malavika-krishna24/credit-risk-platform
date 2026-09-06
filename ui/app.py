"""
ui/app.py

Main Streamlit application for the Credit Risk Intelligence Platform.
Five sections as required: EDA, Risk Prediction, Explainability, Business Rules,
Talk-to-Data chatbot.
"""
import sys
sys.path.insert(0, ".")

import json
import joblib
import pandas as pd
import streamlit as st

from src.ml.predict import predict_risk, prepare_features
from src.ml.explain import explain_prediction, plain_english_summary
from src.utils.config import MODELS_DIR, risk_band

st.set_page_config(page_title="Credit Risk Intelligence Platform", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------------
# Design system — deliberate dark navy/teal palette, IBM Plex Sans throughout.
# Injected once here rather than per-page.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { max-width: 1100px; padding-top: 2rem; }

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; letter-spacing: -0.01em; }

.risk-card {
    padding: 1.5rem 1.75rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    background: #131F30;
    margin-bottom: 1rem;
}
.risk-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.risk-low { background: rgba(76,154,107,0.18); color: #6FCB9A; border: 1px solid #4C9A6B; }
.risk-medium { background: rgba(224,164,88,0.18); color: #F0C079; border: 1px solid #E0A458; }
.risk-high { background: rgba(192,57,43,0.18); color: #F08A7E; border: 1px solid #C0392B; }

.kpi-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1rem;
    font-weight: 500;
    color: #E8EDF3;
}
.kpi-label {
    font-size: 0.85rem;
    color: #8A97AB;
    margin-bottom: 0.2rem;
}
.insight-line { border-left: 3px solid #3FA796; padding-left: 0.9rem; margin: 0.6rem 0; color: #C7D1DD; }
</style>
""", unsafe_allow_html=True)


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


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ◆ Credit Risk Platform")
st.sidebar.caption("AI-powered credit risk intelligence — NeoStats assignment")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Data Exploration (EDA)", "Risk Prediction", "Explainability", "Business Rules", "Talk to Data"],
    label_visibility="collapsed",
)

df = load_features()
model, meta = load_model_and_meta()

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Credit Risk Intelligence Platform")
    st.write(
        "A lightweight, explainable platform for scoring loan-default risk, "
        "built on the Home Credit Default Risk dataset — multi-table feature "
        "engineering, a calibrated ML model, SHAP explainability, derived "
        "business rules, and a natural-language chatbot over the underlying data."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="risk-card"><div class="kpi-label">Applicants</div>'
                     f'<div class="kpi-number">{len(df):,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="risk-card"><div class="kpi-label">Default Rate</div>'
                     f'<div class="kpi-number">{df["TARGET"].mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="risk-card"><div class="kpi-label">Model ROC-AUC</div>'
                     f'<div class="kpi-number">{meta["metrics"]["roc_auc"]:.3f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="risk-card"><div class="kpi-label">Features Used</div>'
                     f'<div class="kpi-number">{len(meta["feature_names"])}</div></div>', unsafe_allow_html=True)

    st.markdown("### How this platform is put together")
    st.markdown("""
    - **Data Exploration** — dataset summary, data quality, and business insights from the raw + joined tables
    - **Risk Prediction** — score a new applicant and get a Low / Medium / High risk band
    - **Explainability** — SHAP-based, plain-English reasons behind any prediction
    - **Business Rules** — auditable IF-THEN rules mined from the model, with measured lift
    - **Talk to Data** — ask questions about the applicant data in plain English
    """)

# ---------------------------------------------------------------------------
# PAGE: EDA
# ---------------------------------------------------------------------------
elif page == "Data Exploration (EDA)":
    st.title("Data Exploration & Insights")
    st.caption("Home Credit application data joined with bureau, previous-application, and POS/cash history.")

    charts = [
        ("notebooks/eda_charts/01_target_distribution.png", "Severe class imbalance: only ~8% of applicants default — this drives the ROC-AUC/PR-AUC evaluation choice and imbalance handling in the ML pipeline."),
        ("notebooks/eda_charts/02_default_by_age.png", "Default rate falls steadily with age, from ~11.4% (20-30) to ~4.9% (60-70)."),
        ("notebooks/eda_charts/03_default_by_income_type.png", "Unemployed / maternity-leave applicants default 4-5x more often than Working or Pensioner applicants."),
        ("notebooks/eda_charts/04_default_by_bureau_overdue.png", "An overdue loan at another bureau nearly doubles default risk (15.9% vs 8.0%)."),
        ("notebooks/eda_charts/05_default_by_prior_refusal.png", "A prior refusal with this lender predicts future default (10.3% vs 7.0%)."),
        ("notebooks/eda_charts/06_credit_income_ratio_dist.png", "Defaulters skew toward higher credit-to-income ratios."),
        ("notebooks/eda_charts/07_missing_data.png", "~50 housing-quality columns are 50-70% missing — structural (tied to housing type), handled via the model's native missing-value support rather than imputation."),
    ]
    for path, caption in charts:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(path, use_container_width=True)
        with col2:
            st.markdown(f'<div class="insight-line">{caption}</div>', unsafe_allow_html=True)
        st.write("")

# ---------------------------------------------------------------------------
# PAGE: Risk Prediction
# ---------------------------------------------------------------------------
elif page == "Risk Prediction":
    st.title("Score a New Applicant")
    st.caption("Fill in the key fields below — the remaining ~130 model features are filled with population "
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
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": -int(age_years * 365.25),
            "YEARS_BIRTH": age_years,
            "DAYS_EMPLOYED": -int(employed_years * 365.25),
            "YEARS_EMPLOYED": employed_years,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "CODE_GENDER": gender,
            "bureau_overdue_loan_count": bureau_overdue,
            "bureau_active_loan_count": bureau_active,
            "prev_refused_count": prev_refused,
            "EXT_SOURCE_1": ext_score, "EXT_SOURCE_2": ext_score, "EXT_SOURCE_3": ext_score,
            "CREDIT_INCOME_RATIO": credit / max(income, 1),
            "ANNUITY_INCOME_RATIO": annuity / max(income, 1),
            "CREDIT_TERM_YEARS": credit / max(annuity, 1) / 12,
        })
        applicant_df = pd.DataFrame([applicant])
        result = predict_risk(applicant_df)
        proba = result.iloc[0]["default_probability"]
        band = result.iloc[0]["risk_band"]

        st.session_state["last_applicant_df"] = applicant_df
        st.session_state["last_prediction"] = (proba, band)

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f'<div class="risk-card">'
                        f'<div class="kpi-label">Default Probability</div>'
                        f'<div class="kpi-number">{proba*100:.1f}%</div>'
                        f'<br>{risk_badge_html(band)}</div>', unsafe_allow_html=True)
        with col2:
            st.write("**What this means:**")
            if band == "Low":
                st.write("This applicant profile is in the Low risk band — default probability is below the "
                         f"{10}% policy threshold. Standard approval workflow applies.")
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
elif page == "Explainability":
    st.title("Explainable AI")
    st.caption("SHAP-based explanation of individual predictions — auditable, feature-level reasoning.")

    if "last_applicant_df" not in st.session_state:
        st.warning("Score an applicant on the **Risk Prediction** page first, then come back here to see why.")
    else:
        applicant_df = st.session_state["last_applicant_df"]
        proba, band = st.session_state["last_prediction"]

        X = prepare_features(applicant_df, meta)
        explanation = explain_prediction(model, X, top_n=8)

        st.markdown(f'{risk_badge_html(band)} &nbsp; **Default probability: {proba*100:.1f}%**',
                    unsafe_allow_html=True)
        st.write("")
        st.write("**Summary:** " + plain_english_summary(explanation))
        st.write("")
        st.write("**Top contributing factors:**")

        for f in explanation["top_features"]:
            direction_icon = "🔺" if f["direction"] == "increased" else "🔻"
            st.markdown(
                f'<div class="risk-card" style="padding:0.9rem 1.2rem; margin-bottom:0.5rem;">'
                f'{direction_icon} <b>{f["feature"]}</b> = {f["value"]} '
                f'&nbsp;&nbsp;<span style="color:#8A97AB;">({f["direction"]} risk, impact {abs(f["shap_contribution"]):.3f})</span>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("### Global Feature Importance")
        st.caption("How each feature affects predictions across the whole applicant population.")
        st.image("documents/eval_charts/shap_summary.png", use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Business Rules
# ---------------------------------------------------------------------------
elif page == "Business Rules":
    st.title("Derived Business Rules")
    st.caption("Plain-English underwriting rules mined from the model's learned patterns, each backed by "
               "measured lift on real validation data — not just model-internal importance.")

    try:
        rules_df = pd.read_csv("documents/derived_business_rules.csv")
        for _, r in rules_df.head(15).iterrows():
            st.markdown(
                f'<div class="risk-card" style="padding:1rem 1.3rem;">'
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
elif page == "Talk to Data":
    st.title("Talk to Your Data")
    st.caption("Ask questions in plain English — answered by an LLM-generated, validated SQL query "
               "against the real applicant database.")

    from src.talk_to_data.nl_to_sql import ask, SAMPLE_QUESTIONS
    from src.utils.config import GROQ_API_KEY

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file (free key at console.groq.com) to use this feature.")
    else:
        st.write("**Try one of these, or type your own question:**")
        cols = st.columns(3)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state["chat_question"] = q

        question = st.text_input("Your question", value=st.session_state.get("chat_question", ""))

        if st.button("Ask", type="primary") and question:
            with st.spinner("Generating query and fetching answer..."):
                result = ask(question)

            if result["sql"]:
                with st.expander("SQL query used"):
                    st.code(result["sql"], language="sql")

            if result["success"]:
                st.success(result["answer"])
                if result["data"] is not None and len(result["data"]) > 0:
                    st.dataframe(result["data"], use_container_width=True)
            else:
                st.warning(result["answer"])
