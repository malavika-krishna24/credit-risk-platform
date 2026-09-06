"""
src/ml/evaluate.py

Standalone evaluation: reproduces the train/val split used during training,
scores the validation set, and saves ROC curve, PR curve, and feature
importance plots to documents/ for the report / presentation.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from src.ml.train import prepare_xy, TARGET_COL
from src.utils.config import MODELS_DIR, BASE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = BASE_DIR / "documents" / "eval_charts"


def main():
    df = pd.read_parquet("data/features_train.parquet")
    X, y, cat_cols = prepare_xy(df)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    proba = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    logger.info(f"Validation ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2E86AB", label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Default Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=110)
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_val, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#C0392B", label=f"PR-AUC = {pr_auc:.3f}")
    plt.axhline(y_val.mean(), color="k", linestyle="--", alpha=0.4, label=f"Baseline ({y_val.mean():.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Default Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curve.png", dpi=110)
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 7))
    plt.barh(importances.index[::-1], importances.values[::-1], color="#4C9A6B")
    plt.title("Top 20 Feature Importances (LightGBM, gain-based)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=110)
    plt.close()

    logger.info(f"Saved evaluation charts to {OUTPUT_DIR}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


if __name__ == "__main__":
    main()
