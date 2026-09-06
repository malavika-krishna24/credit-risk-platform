"""
Central configuration for the Credit Risk Platform.
All modules should import settings from here instead of reading os.environ directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_setting(key: str, default: str = "") -> str:
    """Reads a setting from (in order): the environment / .env file, then
    Streamlit's secrets manager as a fallback. Local dev and Docker use
    .env; the public Streamlit Community Cloud deployment uses st.secrets
    instead (no .env file is ever committed to git), so both paths need to
    work from the same config module without every caller needing to know
    which environment it's running in."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = _get_setting("DB_PATH", str(DATA_DIR / "credit_risk.duckdb"))

# --- LLM ---
GROQ_API_KEY = _get_setting("GROQ_API_KEY", "")
GROQ_MODEL = _get_setting("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Risk bands ---
RISK_LOW_THRESHOLD = float(_get_setting("RISK_LOW_THRESHOLD", "0.10"))
RISK_HIGH_THRESHOLD = float(_get_setting("RISK_HIGH_THRESHOLD", "0.35"))

# --- Logging ---
LOG_LEVEL = _get_setting("LOG_LEVEL", "INFO")

# --- Raw source files expected in data/ ---
RAW_FILES = {
    "app_train": "application_train.csv",
    "app_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


def risk_band(probability: float) -> str:
    """Map a default probability to a business-readable risk band."""
    if probability < RISK_LOW_THRESHOLD:
        return "Low"
    elif probability < RISK_HIGH_THRESHOLD:
        return "Medium"
    return "High"

