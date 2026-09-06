const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in

// ---- Design tokens (matches the app's palette) ----
const BG = "0B1420";
const CARD = "131F30";
const CARD_BORDER = "2A3A50";
const TEXT = "E8EDF3";
const MUTED = "8A97AB";
const TEAL = "3FA796";
const GREEN = "4C9A6B";
const AMBER = "E0A458";
const RED = "C0392B";

const FONT_HEAD = "Calibri";
const FONT_BODY = "Calibri";

const ROOT = __dirname + "/../";

function bgSlide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

function title(s, text, opts = {}) {
  s.addText(text, {
    x: 0.6, y: 0.45, w: 12.1, h: 0.9,
    fontSize: opts.fontSize || 30, bold: true, color: TEXT,
    fontFace: FONT_HEAD, isTextBox: true, align: "left",
  });
}

function kicker(s, text) {
  s.addText(text, {
    x: 0.6, y: 0.18, w: 10, h: 0.35,
    fontSize: 12, color: TEAL, fontFace: FONT_BODY, isTextBox: true, bold: true,
  });
}

function pageNum(s, n) {
  s.addText(String(n), {
    x: 12.7, y: 7.05, w: 0.5, h: 0.35, fontSize: 10, color: MUTED,
    fontFace: FONT_BODY, isTextBox: true, align: "right",
  });
}

// =========================================================
// SLIDE 1 — Title
// =========================================================
{
  const s = bgSlide();
  s.addText("Credit Risk Intelligence Platform", {
    x: 0.9, y: 2.5, w: 11.5, h: 1.3,
    fontSize: 42, bold: true, color: TEXT, fontFace: FONT_HEAD, isTextBox: true,
  });
  s.addText("An AI-powered platform for loan default prediction, explainability, and natural-language data access", {
    x: 0.9, y: 3.65, w: 10.5, h: 0.6,
    fontSize: 16, color: MUTED, fontFace: FONT_BODY, isTextBox: true,
  });
  s.addShape(pres.ShapeType.rect, { x: 0.9, y: 4.35, w: 0.55, h: 0.06, fill: { color: TEAL }, line: { type: "none" } });
  s.addText("NeoStats AI Engineer Internship — Candidate Assignment", {
    x: 0.9, y: 6.7, w: 8, h: 0.4, fontSize: 12, color: MUTED, fontFace: FONT_BODY, isTextBox: true,
  });
}

// =========================================================
// SLIDE 2 — Objective & Approach
// =========================================================
{
  const s = bgSlide();
  kicker(s, "OBJECTIVE");
  title(s, "Building a bank-ready credit risk platform");

  const items = [
    ["Multi-table data", "Joined application, bureau, prior-loan, and monthly repayment history — not just a single table"],
    ["Explainable by design", "Every prediction is backed by SHAP reasoning and auditable business rules, not a black box"],
    ["Talk to the data", "A guardrailed NL-to-SQL chatbot for non-technical stakeholders to explore the data directly"],
    ["Ship it", "Fully Dockerized, single-command deployment"],
  ];
  let y = 1.75;
  items.forEach(([h, d]) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 1.05, rectRadius: 0.08,
      fill: { color: CARD }, line: { color: CARD_BORDER, width: 1 },
    });
    s.addText(h, { x: 0.95, y: y + 0.12, w: 4, h: 0.4, fontSize: 15, bold: true, color: TEAL, fontFace: FONT_HEAD, isTextBox: true });
    s.addText(d, { x: 0.95, y: y + 0.5, w: 11.4, h: 0.45, fontSize: 12.5, color: MUTED, fontFace: FONT_BODY, isTextBox: true });
    y += 1.22;
  });
  pageNum(s, 2);
}

// =========================================================
// SLIDE 3 — Architecture
// =========================================================
{
  const s = bgSlide();
  kicker(s, "ARCHITECTURE");
  title(s, "How the pieces fit together");
  s.addImage({ path: ROOT + "documents/architecture_diagram.png", x: 0.5, y: 1.5, w: 12.3, h: 5.65 });
  pageNum(s, 3);
}

// =========================================================
// SLIDE 4 — Data & Multi-table Feature Engineering
// =========================================================
{
  const s = bgSlide();
  kicker(s, "DATA");
  title(s, "Multi-table feature engineering, not just one CSV");

  s.addText(
    "Most submissions on this dataset use application_train.csv alone. This platform joins five additional tables — aggregated in SQL for speed on tables up to 27M rows — to capture signal a single-table model would miss.",
    { x: 0.6, y: 1.55, w: 12.1, h: 0.7, fontSize: 13.5, color: MUTED, fontFace: FONT_BODY, isTextBox: true }
  );

  const tables = [
    ["applications", "307,511 rows", "Core applicant demographics & loan terms"],
    ["bureau + bureau_balance", "1.7M + 27.3M rows", "External credit history & monthly DPD status"],
    ["previous_applications", "1.7M rows", "This applicant's prior loans with the same lender"],
    ["pos_cash_balance", "10M rows", "Monthly days-past-due (DPD) on cash/POS loans"],
    ["credit_card_balance", "3.8M rows", "Monthly card balance & utilization — new default-risk signal"],
  ];
  let x = 0.5;
  const cardW = 2.42;
  tables.forEach(([name, size, desc]) => {
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.55, w: cardW, h: 3.6, rectRadius: 0.08,
      fill: { color: CARD }, line: { color: TEAL, width: 1 },
    });
    s.addText(name, { x: x + 0.15, y: 2.75, w: cardW - 0.3, h: 0.75, fontSize: 13, bold: true, color: TEXT, fontFace: FONT_HEAD, isTextBox: true });
    s.addText(size, { x: x + 0.15, y: 3.45, w: cardW - 0.3, h: 0.4, fontSize: 10.5, color: TEAL, fontFace: "Consolas", isTextBox: true });
    s.addText(desc, { x: x + 0.15, y: 3.9, w: cardW - 0.3, h: 2.1, fontSize: 10.5, color: MUTED, fontFace: FONT_BODY, isTextBox: true });
    x += cardW + 0.18;
  });

  s.addText("Result: 165 features across 6 tables, feeding a single unified model.", {
    x: 0.6, y: 6.4, w: 12, h: 0.5, fontSize: 13, italic: true, color: TEAL, fontFace: FONT_BODY, isTextBox: true,
  });
  pageNum(s, 4);
}

// =========================================================
// SLIDE 5 — EDA Insights
// =========================================================
{
  const s = bgSlide();
  kicker(s, "EXPLORATORY DATA ANALYSIS");
  title(s, "What the data says before any model is built");

  s.addImage({ path: ROOT + "notebooks/eda_charts/01_target_distribution.png", x: 0.5, y: 1.5, w: 5.8, h: 3.5 });
  s.addImage({ path: ROOT + "notebooks/eda_charts/04_default_by_bureau_overdue.png", x: 6.5, y: 1.5, w: 5.8, h: 3.5 });

  const insights = [
    "Only 8.07% of applicants default — a naive \"approve everyone\" model would be 92% \"accurate\" and useless",
    "An overdue external bureau loan nearly doubles default risk (15.9% vs 8.0%)",
  ];
  s.addText(insights.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < insights.length - 1, color: MUTED, fontSize: 13 } })), {
    x: 0.6, y: 5.2, w: 12, h: 1.7, fontFace: FONT_BODY, isTextBox: true, valign: "top",
  });
  pageNum(s, 5);
}

// =========================================================
// SLIDE 6 — ML Model & Imbalance Strategy
// =========================================================
{
  const s = bgSlide();
  kicker(s, "MACHINE LEARNING");
  title(s, "LightGBM with an evidence-based imbalance strategy");

  // Left: metrics cards
  const metrics = [
    ["ROC-AUC", "0.780"],
    ["PR-AUC", "0.277", ],
    ["Recall @ 0.5", "66.7%"],
    ["Features", "165"],
  ];
  let mx = 0.6;
  metrics.forEach(([label, val]) => {
    s.addShape(pres.ShapeType.roundRect, { x: mx, y: 1.6, w: 2.85, h: 1.3, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARD_BORDER, width: 1 } });
    s.addText(label, { x: mx + 0.18, y: 1.72, w: 2.5, h: 0.35, fontSize: 11, color: MUTED, fontFace: FONT_BODY, isTextBox: true });
    s.addText(val, { x: mx + 0.18, y: 2.05, w: 2.5, h: 0.7, fontSize: 26, bold: true, color: TEXT, fontFace: "Consolas", isTextBox: true });
    mx += 3.05;
  });

  s.addText("Class imbalance (~8% default rate): scale_pos_weight vs SMOTE — tested, not assumed", {
    x: 0.6, y: 3.15, w: 12, h: 0.4, fontSize: 14, bold: true, color: TEAL, fontFace: FONT_HEAD, isTextBox: true,
  });

  // comparison table
  const rows = [
    [{ text: "Approach", options: { bold: true, color: TEXT, fill: { color: CARD } } },
     { text: "Validation ROC-AUC", options: { bold: true, color: TEXT, fill: { color: CARD } } },
     { text: "Notes", options: { bold: true, color: TEXT, fill: { color: CARD } } }],
    [{ text: "scale_pos_weight (production)", options: { color: TEXT } },
     { text: "0.7804", options: { color: GREEN, bold: true } },
     { text: "Simpler, faster, no synthetic data", options: { color: MUTED } }],
    [{ text: "SMOTE (0.5 ratio)", options: { color: TEXT } },
     { text: "0.7792", options: { color: TEXT } },
     { text: "Statistically indistinguishable (difference = 0.001)", options: { color: MUTED } }],
  ];
  s.addTable(rows, {
    x: 0.6, y: 3.65, w: 12.1, h: 1.4, fontSize: 12.5, fontFace: FONT_BODY,
    border: { type: "solid", color: CARD_BORDER, pt: 1 },
    fill: { color: BG }, autoPage: false, colW: [4.5, 3, 4.6],
  });

  s.addText("Evaluated on ROC-AUC / PR-AUC, not accuracy — accuracy is misleading under 8% class imbalance.", {
    x: 0.6, y: 5.3, w: 12, h: 0.4, fontSize: 12.5, italic: true, color: MUTED, fontFace: FONT_BODY, isTextBox: true,
  });

  s.addImage({ path: ROOT + "documents/eval_charts/feature_importance.png", x: 2.6, y: 5.68, w: 8, h: 1.62 });
  pageNum(s, 6);
}

// =========================================================
// SLIDE 7 — Explainability
// =========================================================
{
  const s = bgSlide();
  kicker(s, "EXPLAINABLE AI");
  title(s, "SHAP: auditable reasons behind every prediction");
  s.addImage({ path: ROOT + "documents/screenshots/04_explainability.png", x: 3.9, y: 1.4, w: 6.6, h: 5.83 });

  const notes = [
    "TreeExplainer — exact SHAP values for gradient-boosted trees, not an approximation",
    "Per-applicant: top features, direction, and a plain-English summary",
    "Global: beeswarm plot across a 2,000-applicant sample",
    "Example: correctly flagged 22 active bureau loans + 70.7% refusal rate on a genuinely-defaulted applicant (p=0.955)",
  ];
  s.addText(notes.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < notes.length - 1, color: MUTED, fontSize: 12.5, paraSpaceAfter: 10 } })), {
    x: 0.5, y: 1.9, w: 3.2, h: 5, fontFace: FONT_BODY, isTextBox: true, valign: "top",
  });
  pageNum(s, 7);
}

// =========================================================
// SLIDE 8 — Business Rules
// =========================================================
{
  const s = bgSlide();
  kicker(s, "BUSINESS RULES");
  title(s, "Bridging ML output and credit policy");
  s.addImage({ path: ROOT + "documents/screenshots/05_business_rules.png", x: 3.9, y: 1.4, w: 6.6, h: 5.83 });

  const notes = [
    "Bins top-25 model features and surfaces any bin where default rate is materially above baseline",
    "Every rule backed by measured lift + support (n), not just model-internal importance",
    "Auditable by a policy team independent of trusting the ML model",
    "Example: Low-skill Laborers default at 2.12x baseline (n=2,093)",
  ];
  s.addText(notes.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < notes.length - 1, color: MUTED, fontSize: 12.5, paraSpaceAfter: 10 } })), {
    x: 0.5, y: 1.9, w: 3.2, h: 5, fontFace: FONT_BODY, isTextBox: true, valign: "top",
  });
  pageNum(s, 8);
}

// =========================================================
// SLIDE 9 — Talk to Data / Hallucination control
// =========================================================
{
  const s = bgSlide();
  kicker(s, "TALK-TO-DATA");
  title(s, "NL-to-SQL with guardrails enforced in code");
  s.addImage({ path: ROOT + "documents/screenshots/06_talk_to_data.png", x: 3.9, y: 1.4, w: 6.6, h: 5.83 });

  const notes = [
    "Two-stage LLM calls: SQL generation (temp 0.1) \u2192 grounded answer summarization from real returned rows",
    "SQL validator blocks: DDL/DML keywords, chained statements, hallucinated tables/columns",
    "Auto row-limiting + read-only DB connection as a second defense layer",
    "Tested against real adversarial inputs (DROP TABLE, injection, fake tables) \u2014 all correctly blocked",
    "7 verified working query patterns (exceeds the required 5)",
  ];
  s.addText(notes.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < notes.length - 1, color: MUTED, fontSize: 11.8, paraSpaceAfter: 8 } })), {
    x: 0.5, y: 1.75, w: 3.2, h: 5.3, fontFace: FONT_BODY, isTextBox: true, valign: "top",
  });
  pageNum(s, 9);
}

// =========================================================
// SLIDE 10 — UI Walkthrough
// =========================================================
{
  const s = bgSlide();
  kicker(s, "USER INTERFACE");
  title(s, "One platform, five sections, real data throughout");
  s.addImage({ path: ROOT + "documents/screenshots/01_overview.png", x: 0.5, y: 1.5, w: 6, h: 4.17 });
  s.addImage({ path: ROOT + "documents/screenshots/03b_risk_prediction_result.png", x: 6.8, y: 1.5, w: 6, h: 4.17 });
  s.addText("Overview (KPIs from real data)                                              Risk Prediction (live scoring + risk band)", {
    x: 0.5, y: 5.75, w: 12.3, h: 0.4, fontSize: 11.5, color: MUTED, fontFace: FONT_BODY, isTextBox: true,
  });
  s.addText("Streamlit \u2022 dark navy/teal design system \u2022 IBM Plex Sans \u2022 responsive KPI cards", {
    x: 0.5, y: 6.5, w: 12, h: 0.4, fontSize: 12.5, italic: true, color: TEAL, fontFace: FONT_BODY, isTextBox: true,
  });
  pageNum(s, 10);
}

// =========================================================
// SLIDE 11 — Deployment
// =========================================================
{
  const s = bgSlide();
  kicker(s, "DEPLOYMENT");
  title(s, "One command to run the full platform");

  const steps = [
    ["1", "Clone & configure", "git clone the repo, copy .env.example \u2192 .env, add a free Groq API key"],
    ["2", "Prep data & model", "Download 5 Kaggle CSVs \u2192 run loader, preprocessor, train, evaluate, rules scripts"],
    ["3", "docker-compose up", "Builds the image and launches Streamlit on port 8501 \u2014 single command"],
  ];
  let y = 1.8;
  steps.forEach(([num, h, d]) => {
    s.addShape(pres.ShapeType.ellipse, { x: 0.6, y, w: 0.6, h: 0.6, fill: { color: TEAL }, line: { type: "none" } });
    s.addText(num, { x: 0.6, y, w: 0.6, h: 0.6, align: "center", valign: "middle", fontSize: 18, bold: true, color: BG, fontFace: FONT_HEAD, isTextBox: true });
    s.addText(h, { x: 1.45, y: y - 0.05, w: 5, h: 0.4, fontSize: 15, bold: true, color: TEXT, fontFace: FONT_HEAD, isTextBox: true });
    s.addText(d, { x: 1.45, y: y + 0.32, w: 10.8, h: 0.6, fontSize: 12.5, color: MUTED, fontFace: FONT_BODY, isTextBox: true });
    y += 1.15;
  });

  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.4, w: 12.1, h: 1.4, rectRadius: 0.08, fill: { color: CARD }, line: { color: TEAL, width: 1 } });
  s.addText("Dockerfile \u2022 docker-compose.yml \u2022 .env.example \u2014 all in the repo root, per the required structure", {
    x: 0.95, y: 5.6, w: 11.4, h: 0.4, fontSize: 13, bold: true, color: TEXT, fontFace: FONT_HEAD, isTextBox: true,
  });
  s.addText("github.com/malavika-krishna24/credit-risk-platform", {
    x: 0.95, y: 6.0, w: 11.4, h: 0.4, fontSize: 12.5, color: TEAL, fontFace: "Consolas", isTextBox: true,
  });
  pageNum(s, 11);
}

// =========================================================
// SLIDE 12 — Known Limitations
// =========================================================
{
  const s = bgSlide();
  kicker(s, "HONEST SCOPE");
  title(s, "Known limitations & what's next");

  const items = [
    ["installments_payments.csv not joined", "Repayment-timing signal is already substantially covered by POS_CASH_balance and credit_card_balance's own DPD tracking"],
    ["Risk-band thresholds are defaults", "Low/Medium/High cutoffs should be tuned against a real bank's approval/loss economics in production"],
    ["Prediction form uses ~12 key fields", "Remaining ~153 features filled from population medians/modes \u2014 a deliberate UX tradeoff for a demo interface"],
    ["No monitoring/retraining pipeline", "A production deployment needs drift detection and scheduled retraining \u2014 out of scope here"],
  ];
  let y = 1.7;
  items.forEach(([h, d]) => {
    s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h: 1.15, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARD_BORDER, width: 1 } });
    s.addText(h, { x: 0.95, y: y + 0.12, w: 11.4, h: 0.4, fontSize: 14, bold: true, color: AMBER, fontFace: FONT_HEAD, isTextBox: true });
    s.addText(d, { x: 0.95, y: y + 0.53, w: 11.4, h: 0.55, fontSize: 12, color: MUTED, fontFace: FONT_BODY, isTextBox: true });
    y += 1.32;
  });
  pageNum(s, 12);
}

// =========================================================
// SLIDE 13 — Closing
// =========================================================
{
  const s = bgSlide();
  s.addText("Thank you", {
    x: 0.9, y: 2.9, w: 10, h: 1, fontSize: 40, bold: true, color: TEXT, fontFace: FONT_HEAD, isTextBox: true,
  });
  s.addText("Credit Risk Intelligence Platform \u2014 built end-to-end with real data, tested at every stage", {
    x: 0.9, y: 3.8, w: 10, h: 0.5, fontSize: 15, color: MUTED, fontFace: FONT_BODY, isTextBox: true,
  });
  s.addShape(pres.ShapeType.rect, { x: 0.9, y: 4.4, w: 0.55, h: 0.06, fill: { color: TEAL }, line: { type: "none" } });
  s.addText("github.com/malavika-krishna24/credit-risk-platform", {
    x: 0.9, y: 6.7, w: 8, h: 0.4, fontSize: 12, color: TEAL, fontFace: "Consolas", isTextBox: true,
  });
}

pres.writeFile({ fileName: ROOT + "documents/Credit_Risk_Platform_Presentation.pptx" }).then(() => {
  console.log("Deck written.");
});
