"""
src/ml/predict.py

Inference module. Loads the saved LightGBM model + metadata and scores new
applicants, returning both a raw default probability and a business-readable
risk band (Low / Medium / High), as required by the assignment.
"""
import json

import joblib
import pandas as pd

from src.utils.config import MODELS_DIR, risk_band
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_metadata = None


def load_model():
    """Lazily load and cache the model + metadata (avoids re-reading disk on
    every prediction call, which matters for the Streamlit UI's interactivity)."""
    global _model, _metadata
    if _model is None:
        _model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
        with open(MODELS_DIR / "model_metadata.json") as f:
            _metadata = json.load(f)
        logger.info("Model + metadata loaded.")
    return _model, _metadata


def prepare_features(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Align an arbitrary input dataframe to exactly the feature set/dtypes the
    model was trained on — filling any missing expected columns with NaN and
    dropping anything extra, so inference never breaks on schema drift."""
    feature_names = metadata["feature_names"]
    cat_cols = metadata["categorical_features"]

    X = df.reindex(columns=feature_names)
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def predict_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Score a batch of applicants. Returns a dataframe with SK_ID_CURR (if
    present), default_probability, and risk_band."""
    model, metadata = load_model()
    ids = df["SK_ID_CURR"] if "SK_ID_CURR" in df.columns else pd.Series(range(len(df)), name="row_id")

    X = prepare_features(df, metadata)
    proba = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        ids.name: ids.values,
        "default_probability": proba,
    })
    result["risk_band"] = result["default_probability"].apply(risk_band)
    return result


def predict_single(applicant: dict) -> dict:
    """Convenience wrapper for scoring a single applicant (e.g. from a UI form)."""
    df = pd.DataFrame([applicant])
    result = predict_risk(df)
    row = result.iloc[0]
    return {
        "default_probability": float(row["default_probability"]),
        "risk_band": row["risk_band"],
    }


if __name__ == "__main__":
    # Smoke test: score a slice of the training set (with TARGET dropped, as at
    # real inference time we wouldn't have it) to confirm the pipeline works end-to-end.
    df = pd.read_parquet("data/features_train.parquet").drop(columns=["TARGET"]).head(10)
    result = predict_risk(df)
    print(result)
