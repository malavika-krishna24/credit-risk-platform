"""
Central configuration for the Credit Risk Platform.
All modules should import settings from here instead of reading os.environ directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "credit_risk.duckdb"))

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Risk bands ---
RISK_LOW_THRESHOLD = float(os.getenv("RISK_LOW_THRESHOLD", 0.10))
RISK_HIGH_THRESHOLD = float(os.getenv("RISK_HIGH_THRESHOLD", 0.35))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

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
