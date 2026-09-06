"""
src/ml/train.py

Trains a LightGBM binary classifier to predict loan default probability.

Class imbalance strategy (documented in detail in README.md):
  - Primary approach: `scale_pos_weight` (native to LightGBM) — reweights the loss
    function so the ~8% minority (default) class contributes proportionally more,
    without synthesizing data or discarding majority-class rows. This is generally
    the better choice for gradient-boosted trees vs SMOTE, which can introduce
    unrealistic synthetic points in high-dimensional, partly-categorical data.
  - We ALSO run a SMOTE-oversampled comparison run (see `run_smote_comparison`)
    purely to justify this choice with evidence rather than assumption, and log
    both results so the tradeoff is visible in the README.

Evaluation: ROC-AUC and PR-AUC (NOT accuracy — with an 8% positive rate, a
model predicting "no default" for everyone would score 92% accuracy while being
useless). PR-AUC is emphasized because it is more informative than ROC-AUC under
heavy class imbalance.
"""
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
DROP_COLS = {TARGET_COL, ID_COL}


def prepare_xy(df: pd.DataFrame):
    """Split into X/y, cast object columns to pandas 'category' dtype so LightGBM
    can use its native (efficient, no one-hot-explosion) categorical handling."""
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")

    return X, y, cat_cols


def train_lightgbm(X_train, y_train, X_val, y_val, cat_cols, scale_pos_weight=None, params_override=None):
    if scale_pos_weight is None:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = dict(
        objective="binary",
        metric=["auc", "average_precision"],
        learning_rate=0.03,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if params_override:
        params.update(params_override)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model


def evaluate_model(model, X_val, y_val) -> dict:
    proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)

    preds_05 = (proba >= 0.5).astype(int)
    report = classification_report(y_val, preds_05, output_dict=True)
    cm = confusion_matrix(y_val, preds_05).tolist()

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "classification_report_at_0.5": report,
        "confusion_matrix_at_0.5": cm,
    }
    logger.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return metrics


def run_smote_comparison(X_train, y_train, X_val, y_val, cat_cols) -> float:
    """Train a SMOTE-oversampled variant purely for comparison against the
    scale_pos_weight approach. SMOTE needs numeric-only input, so categoricals
    are temporarily label-encoded for this comparison run only (the production
    model does NOT use this path)."""
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import OrdinalEncoder

    X_enc = X_train.copy()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        X_enc[cat_cols] = encoder.fit_transform(X_enc[cat_cols].astype(str))
    X_enc = X_enc.fillna(-999)

    X_val_enc = X_val.copy()
    if cat_cols:
        X_val_enc[cat_cols] = encoder.transform(X_val_enc[cat_cols].astype(str))
    X_val_enc = X_val_enc.fillna(-999)

    logger.info("Running SMOTE oversampling for comparison ...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # bring minority to 50% of majority, not full 1:1
    X_res, y_res = smote.fit_resample(X_enc, y_train)
    logger.info(f"SMOTE resampled train shape: {X_res.shape} (from {X_enc.shape})")

    smote_model = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=48,
        n_estimators=500, random_state=42, n_jobs=-1, verbosity=-1,
    )
    smote_model.fit(X_res, y_res)
    proba = smote_model.predict_proba(X_val_enc)[:, 1]
    smote_roc_auc = roc_auc_score(y_val, proba)
    logger.info(f"SMOTE-variant ROC-AUC: {smote_roc_auc:.4f}")
    return float(smote_roc_auc)


def save_artifacts(model, cat_cols, feature_names, metrics, smote_comparison_auc=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")

    metadata = {
        "feature_names": feature_names,
        "categorical_features": cat_cols,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "smote_comparison_roc_auc": smote_comparison_auc,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model + metadata to {MODELS_DIR}")


def main():
    df = pd.read_parquet(Path("data") / "features_train.parquet")
    logger.info(f"Loaded training data: {df.shape}")

    X, y, cat_cols = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Train default rate: {y_train.mean():.4f}, Val default rate: {y_val.mean():.4f}")

    model = train_lightgbm(X_train, y_train, X_val, y_val, cat_cols)
    metrics = evaluate_model(model, X_val, y_val)

    smote_auc = None
    try:
        smote_auc = run_smote_comparison(X_train, y_train, X_val, y_val, cat_cols)
        logger.info(
            f"Imbalance strategy comparison — scale_pos_weight ROC-AUC: {metrics['roc_auc']:.4f} "
            f"vs SMOTE ROC-AUC: {smote_auc:.4f}"
        )
    except Exception as e:
        logger.warning(f"SMOTE comparison skipped due to error: {e}")

    save_artifacts(model, cat_cols, list(X.columns), metrics, smote_auc)
    return model, metrics


if __name__ == "__main__":
    main()
