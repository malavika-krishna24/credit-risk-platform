"""
src/ml/explain.py

Explainable AI module using SHAP (TreeExplainer, exact for gradient-boosted
trees). Provides two things, as required by the assignment:
  1. Per-applicant explanation: which features drove THIS prediction, and in
     which direction, translated into a plain-English sentence for non-technical
     users (e.g. loan officers, auditors).
  2. Global explanation: which features matter most across the whole model
     (feature_importance.png already covers gain-based importance; this adds
     SHAP's more rigorous, prediction-consistent view).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)

_explainer = None


def get_explainer(model):
    global _explainer
    if _explainer is None:
        logger.info("Building SHAP TreeExplainer ...")
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_prediction(model, X_row: pd.DataFrame, top_n: int = 5) -> dict:
    """Explain a single applicant's prediction. X_row must be a 1-row dataframe
    with the exact feature columns/dtypes the model was trained on."""
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(X_row)

    # LightGBM binary classifier via TreeExplainer returns a single array for
    # the positive class in recent shap versions; handle both cases defensively.
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0] if shap_values.ndim == 2 else shap_values[0, :, 1]

    contributions = pd.Series(values, index=X_row.columns).sort_values(key=abs, ascending=False)
    top_features = contributions.head(top_n)

    explanations = []
    for feat, val in top_features.items():
        direction = "increased" if val > 0 else "decreased"
        raw_value = X_row.iloc[0][feat]
        explanations.append({
            "feature": feat,
            "value": raw_value if not pd.isna(raw_value) else "missing",
            "shap_contribution": float(val),
            "direction": direction,
            "sentence": f"{feat} = {raw_value} {direction} the predicted risk"
                        f" (impact: {abs(val):.3f})",
        })

    return {
        "top_features": explanations,
        "base_value": float(explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray))
                             else explainer.expected_value[1]),
    }


def plain_english_summary(explanation: dict) -> str:
    """Turn the structured explanation into a short paragraph a loan officer or
    auditor can read without any ML background."""
    risk_raisers = [f for f in explanation["top_features"] if f["direction"] == "increased"]
    risk_reducers = [f for f in explanation["top_features"] if f["direction"] == "decreased"]

    parts = []
    if risk_raisers:
        names = ", ".join(f["feature"].replace("_", " ").lower() for f in risk_raisers[:3])
        parts.append(f"Risk was pushed UP mainly by: {names}.")
    if risk_reducers:
        names = ", ".join(f["feature"].replace("_", " ").lower() for f in risk_reducers[:3])
        parts.append(f"Risk was pulled DOWN mainly by: {names}.")

    return " ".join(parts) if parts else "No dominant risk drivers identified for this applicant."


def global_summary_plot(model, X_sample: pd.DataFrame, output_path: str, max_display: int = 15):
    """Save a SHAP beeswarm summary plot showing global feature impact direction
    and magnitude across a sample of applicants."""
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        values = shap_values[1]
    elif shap_values.ndim == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    plt.figure()
    shap.summary_plot(values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_path}")
