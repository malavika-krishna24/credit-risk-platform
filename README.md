# Credit Risk Intelligence Platform

An AI-powered credit risk platform built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset — multi-table feature engineering, an explainable ML model, auditable business rules, and a natural-language chatbot over the underlying applicant data.

Built for the NeoStats AI Engineer Internship candidate assignment.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (ui/app.py)                     │
│   Overview │ EDA │ Risk Prediction │ Explainability │ Rules │ Chat   │
└───────────┬─────────────┬──────────────┬──────────────┬─────────────┘
            │             │              │              │
            ▼             ▼              ▼              ▼
    ┌───────────┐ ┌───────────────┐ ┌──────────┐ ┌──────────────────┐
    │  EDA data │ │  ML Pipeline  │ │  SHAP    │ │  Talk-to-Data     │
    │ (parquet) │ │ (LightGBM)    │ │ Explain  │ │  (Groq LLM)       │
    └───────────┘ └───────┬───────┘ └────┬─────┘ └─────────┬─────────┘
                           │              │                 │
                           ▼              ▼                 ▼
                  ┌─────────────────────────────────────────────┐
                  │      SQL Guardrail (validate → execute)       │
                  │         src/talk_to_data/query_runner.py       │
                  └───────────────────────┬───────────────────────┘
                                           ▼
                  ┌─────────────────────────────────────────────┐
                  │        DuckDB (data/credit_risk.duckdb)        │
                  │  applications │ bureau │ previous_applications  │
                  │            │ pos_cash_balance │                 │
                  └─────────────────────────────────────────────┘
                                           ▲
                                           │
                  ┌─────────────────────────────────────────────┐
                  │  Raw Kaggle CSVs → src/data/loader.py           │
                  │           → src/data/preprocessor.py            │
                  │        (multi-table aggregation, in SQL)        │
                  └─────────────────────────────────────────────┘
```

**Data flow:** raw CSVs are loaded into DuckDB (`src/data/loader.py`), which serves two consumers: (1) the feature-engineering pipeline (`src/data/preprocessor.py`), which aggregates `bureau`, `previous_application`, and `POS_CASH_balance` in SQL and joins them onto the main application table for model training; and (2) the talk-to-data chatbot, which queries the same DuckDB tables directly through a validated SQL layer — so the chatbot is always answering from the same ground truth as the model.

---

## 2. What's in this repo

| # | Module | Files |
|---|--------|-------|
| 1 | Data loading & feature engineering | `src/data/loader.py`, `src/data/preprocessor.py` |
| 2 | EDA | `notebooks/eda.ipynb`, `notebooks/eda.py`, `notebooks/eda_charts/` |
| 3 | ML training & evaluation | `src/ml/train.py`, `src/ml/predict.py`, `src/ml/evaluate.py` |
| 4 | Explainable AI (SHAP) | `src/ml/explain.py` |
| 5 | Business rule derivation | `src/ml/rules.py` |
| 6 | Talk-to-Data (NL→SQL) | `src/talk_to_data/prompt_templates.py`, `nl_to_sql.py`, `query_runner.py` |
| 7 | UI | `ui/app.py` (Streamlit, 5 sections) |
| 8 | Deployment | `Dockerfile`, `docker-compose.yml`, `.env.example` |

---

## 3. Setup & Run Instructions

### Option A — Docker (recommended, matches the assignment's evaluation method)

```bash
# 1. Clone the repo
git clone <this-repo-url>
cd credit-risk-platform

# 2. Download the dataset files from Kaggle (see section 4 below) into data/

# 3. Copy the env template and add your free Groq API key
cp .env.example .env
# edit .env and set GROQ_API_KEY (get one free at https://console.groq.com)

# 4. Build the feature table, train the model, and generate SHAP/rules artifacts
#    (run once, locally, before first docker-compose up — see section 5)
pip install -r requirements.txt
python -m src.data.loader
python scripts/build_features.py
python -m src.ml.train
python -m src.ml.evaluate
python -m src.ml.rules

# 5. Launch the full platform
docker-compose up --build
```

Then open **http://localhost:8501**.

### Option B — Local (no Docker)

```bash
pip install -r requirements.txt
# ... same data/model prep steps as above ...
streamlit run ui/app.py
```

### Verifying the chatbot works live

```bash
python scripts/test_chatbot_live.py
```

This runs all 6 sample questions against the real Groq API and prints the generated SQL + answer for each — useful as a first check after setup.

---

## 4. Dataset

Download from [Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place these files in `data/` (never committed to git — see `.gitignore`):

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`

**Note on scope:** `bureau_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`, and `sample_submission.csv` from the full Kaggle download are intentionally not used. The five tables above already cover application data, external bureau history, prior-loan history, and monthly repayment/DPD behavior — the four categories of signal that matter most for default prediction. Adding the remaining tables (mostly `bureau_balance` and `credit_card_balance`) is a natural extension; see "Known Limitations" below.

---

## 5. Model Selection & Class Imbalance Strategy

**Model:** LightGBM (gradient-boosted trees). Chosen over logistic regression / random forest because it natively handles missing values (relevant here — ~50 housing-quality columns are 50-70% missing) and categorical features (no one-hot encoding blow-up across features like `ORGANIZATION_TYPE` with 50+ categories), while training fast enough on 300K rows × 150+ features to iterate quickly within the assignment timeline.

**Class imbalance** (~8.07% default rate) is handled primarily via **`scale_pos_weight`** — LightGBM's native loss-reweighting, which biases the model toward the minority class without synthesizing or discarding data.

We validated this choice empirically rather than assuming it: a SMOTE-oversampled variant was trained for direct comparison.

| Approach | Validation ROC-AUC |
|---|---|
| `scale_pos_weight` (production model) | **0.7782** |
| SMOTE (0.5 sampling ratio) | 0.7793 |

The two are statistically indistinguishable (Δ = 0.001), so we kept `scale_pos_weight` as the production approach — it's simpler, faster to train (no resampling step), and avoids the risk SMOTE carries in high-dimensional, partly-categorical data: synthetic points interpolated between real applicants aren't guaranteed to be realistic in a space with 16 categorical columns. This comparison is reproducible via `src/ml/train.py::run_smote_comparison`.

We also evaluate on **ROC-AUC and PR-AUC**, not accuracy — with an 8% positive rate, a model that predicts "no default" for every applicant scores 92% accuracy while being completely useless to a bank. PR-AUC (0.273, vs a 0.081 no-skill baseline) is emphasized because it's more informative than ROC-AUC under heavy class imbalance.

---

## 6. Evaluation Metrics & Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.7782 |
| PR-AUC | 0.2729 (baseline/no-skill: 0.0807) |
| Recall @ 0.5 threshold (class 1 / default) | 66.2% |
| Precision @ 0.5 threshold (class 1 / default) | 19.2% |

At the default 0.5 threshold the model favors **recall over precision** for the default class — a deliberate consequence of `scale_pos_weight`, and the right trade-off for a bank: missing a defaulter (false negative) is typically costlier than flagging a safe applicant for review (false positive). Rather than force a hard cutoff, the platform exposes calibrated probabilities and three **risk bands** — Low (<10%), Medium (10-35%), High (>35%), configurable via `.env` — so a credit team can set their own threshold per policy.

See `documents/eval_charts/roc_curve.png`, `pr_curve.png`, and `feature_importance.png` for the full evaluation visuals, generated by `src/ml/evaluate.py`.

**Feature importance validates the multi-table approach**: engineered features from `bureau` (`bureau_total_credit`, `bureau_debt_credit_ratio`, `bureau_max_overdue_amt`), `previous_application` (`prev_avg_annuity`, `prev_avg_credit_amt`), and `pos_cash_balance` (`pos_months_count`) all rank in the top 20 most important features — alongside the dataset's well-known `EXT_SOURCE_1/2/3` external credit scores.

---

## 7. Explainable AI

`src/ml/explain.py` uses SHAP's `TreeExplainer` (exact for gradient-boosted trees, not an approximation) to explain both individual predictions and global model behavior:

- **Per-applicant**: top contributing features, direction (increased/decreased risk), and a plain-English summary sentence — shown in the UI's Explainability tab right after scoring an applicant.
- **Global**: a SHAP beeswarm summary plot (`documents/eval_charts/shap_summary.png`) showing feature impact direction and magnitude across a 2,000-applicant sample.

Example (real output on a genuinely-defaulted applicant, predicted probability 0.955): the model correctly identified 22 active bureau loans, a 70.7% prior-refusal rate, and 2 overdue bureau loans as the dominant risk drivers — all sensible, auditable reasons a credit officer could act on.

---

## 8. Business Rule Derivation

`src/ml/rules.py` bridges ML output and credit policy: it takes the model's top 25 most important features, bins numeric features into quantiles (and groups categorical features by value), and surfaces any bin/category where the empirical default rate is materially above baseline (configurable lift and minimum-support thresholds — default 1.3x lift, 200 minimum applicants).

This produces **auditable, plain-English rules backed by measured lift on real validation data** — not just model-internal feature importance — so a credit policy team can review and adopt them independently of trusting the ML model as a black box.

Sample output (from `documents/derived_business_rules.csv`):

```
- If OCCUPATION_TYPE == 'Low-skill Laborers', default rate is 17.2% vs 8.1% baseline (2.12x higher risk, n=2,093).
- If EXT_SOURCE_3 in [-0.00, 0.37], default rate is 15.1% vs 8.1% baseline (1.87x higher risk, n=61,993).
- If bureau_debt_credit_ratio in [0.49, 7.79], default rate is 11.5% vs 8.1% baseline (1.43x higher risk, n=63,789).
```

---

## 9. Talk-to-Data: Prompt Engineering & Hallucination Control

**LLM provider:** [Groq](https://console.groq.com) (Llama 3.3 70B) — chosen over OpenAI/Anthropic because it has a genuinely free tier (no card required), which matters since an evaluator running this repo shouldn't need to spend their own money, and its inference speed keeps the chatbot feeling responsive.

**Two-stage prompting** (`src/talk_to_data/prompt_templates.py`):
1. **SQL generation** — the LLM is given an explicit, exhaustive schema description (only 4 real tables, exact column names/types) and few-shot examples, and instructed to output *only* SQL, `LIMIT`-bounded, or a `NO_QUERY:` refusal if the question is unanswerable from the schema. Temperature 0.1 for consistency.
2. **Answer summarization** — a second, separate LLM call is given the *actual returned rows* (not the question alone) and asked to summarize only what's in that data. This grounds the answer and prevents the model from inventing numbers not present in the query result.

**Hallucination/injection control is enforced in code, not just prompted for** (`src/talk_to_data/query_runner.py`):
- Only `SELECT`/`WITH` statements accepted — any DDL/DML keyword (`DROP`, `INSERT`, `UPDATE`, `DELETE`, `ALTER`, etc.) is rejected before execution, even if the LLM ever tries to "help" by modifying data.
- Chained statements (`; DROP TABLE ...`) are rejected.
- Every table reference is checked against an explicit allow-list — a hallucinated table/column name is caught and rejected with a clear reason, rather than hitting the database and surfacing a confusing raw error.
- Queries without an explicit `LIMIT` are auto-limited to 500 rows; any `LIMIT` above 500 is clamped.
- The DB connection is opened **read-only**, as a second independent layer of defense beyond the SQL validator.

All of the above was tested against real adversarial inputs (`DROP TABLE`, chained-statement injection, a hallucinated table name) during development — all were correctly blocked while legitimate queries passed through unmodified.

**Token optimization:** the schema description is written once and reused for every request (no re-fetching or re-describing schema per query); `max_tokens` is capped per call (400 for SQL generation, 300 for answer summarization) since both outputs are naturally short.

**Sample working queries** (6, exceeding the required 5 — see `SAMPLE_QUESTIONS` in `nl_to_sql.py`, all validated against the real database):
1. "What is the average income of applicants who defaulted vs those who didn't?"
2. "How many applicants have more than 2 overdue bureau loans?"
3. "What's the default rate for applicants with a college education?"
4. "Show me the top 10 organization types by average credit amount."
5. "How many applicants were previously refused a loan by this lender?"
6. "What is the average credit amount for female vs male applicants?"

---

## 10. Known Limitations & Possible Improvements

- **Tables not yet joined**: `bureau_balance.csv` (monthly bureau credit-line snapshots) and `credit_card_balance.csv` (monthly card utilization) were left out of this submission's scope to keep the pipeline manageable within the assignment timeline — both would add further repayment-behavior signal on top of what `bureau` and `pos_cash_balance` already provide.
- **Risk-band thresholds** (Low <10%, Medium 10-35%, High >35%) are configured defaults, not derived from a cost-sensitive optimization against a specific bank's approval/loss economics — in production these should be tuned against real business costs of false positives vs false negatives.
- **The Risk Prediction UI form** exposes ~12 key fields and fills the rest from population medians/modes, rather than all 150+ model features — a deliberate UX tradeoff for a demo interface; a production underwriting system would pull the full applicant record automatically rather than via manual form entry.
- **No model monitoring/retraining pipeline** — a production deployment would need drift detection and a scheduled retraining job, out of scope for this assignment.
- **SQL validator is allow-list based**, not a full SQL parser — it uses regex-based table extraction, which is robust for the LLM's actual output patterns (verified via prompt few-shot examples) but is not a formally verified parser. A production system handling truly adversarial input (vs. an LLM that's prompted to behave) would benefit from a proper SQL AST parser (e.g. `sqlglot`).

---

## 11. Tech Stack

| Component | Choice | Why |
|---|---|---|
| ML model | LightGBM | Native categorical + missing-value handling, fast on this scale |
| Explainability | SHAP (TreeExplainer) | Exact (not approximate) for tree models |
| Database | DuckDB | Fast in-process SQL on CSV-scale data, doubles as the chatbot's query target |
| LLM | Groq (Llama 3.3 70B) | Free tier, fast inference, no cost burden on the evaluator |
| UI | Streamlit | Fast to build a clean multi-section app; matches "lightweight platform" framing |
| Deployment | Docker + Docker Compose | Single-command reproducible run, as required |

---

## 12. Local Development Workflow

### Full local setup (Python + Node)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Add your dataset
#    Download from Kaggle (see section 4) and place these 5 files in data/:
#    application_train.csv, application_test.csv, bureau.csv,
#    previous_application.csv, POS_CASH_balance.csv

# 4. Add your Groq API key
cp .env.example .env
# edit .env, set GROQ_API_KEY (free key at https://console.groq.com)

# 5. Build the database, features, model, and derived artifacts (run once)
python -m src.data.loader
python scripts/build_features.py
python -m src.ml.train
python -m src.ml.evaluate
python -m src.ml.rules

# 6. Run the dashboard
streamlit run ui/app.py
```

Open **http://localhost:8501**. Streamlit auto-reloads on file save — edit `ui/app.py` and the browser tab refreshes automatically (or press "Rerun" if auto-reload is off).

### Modifying the dashboard

`ui/app.py` is a single file with 5 sections, each in its own `elif page == "..."` block. The design tokens (colors, fonts) are the CSS block injected near the top — change `#0B1420`, `#3FA796`, etc. there to re-theme everything at once. After any model/feature changes, re-run step 5 above so the UI picks up the new artifacts.

### Modifying the presentation

The deck is generated from code, not hand-edited — this makes it easy to update consistently as the project changes.

```bash
npm install                          # installs pptxgenjs (only needed once)
node scripts/build_presentation.js   # regenerates documents/Credit_Risk_Platform_Presentation.pptx
```

Edit `scripts/build_presentation.js` to change slide content, then re-run. Each slide is a clearly-labeled block (`SLIDE 1 — Title`, `SLIDE 2 — Objective`, etc.) — colors and fonts are the same design-token constants at the top of the file as the dashboard, so the two stay visually consistent.

To convert to PDF after edits (matches the submission requirement):
```bash
soffice --headless --convert-to pdf documents/Credit_Risk_Platform_Presentation.pptx --outdir documents/
```
(Or just open the `.pptx` in PowerPoint/Keynote/Google Slides and export to PDF manually — it's a normal editable file.)
