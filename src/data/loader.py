"""
src/data/loader.py

Loads the raw Home Credit CSVs from data/ into a single DuckDB database file.
This DB serves two purposes:
  1. Source for feature engineering (src/data/preprocessor.py)
  2. Backing store for the NL-to-SQL talk-to-data chatbot (src/talk_to_data/)
"""
import duckdb
import pandas as pd
from pathlib import Path

from src.utils.config import DATA_DIR, DB_PATH, RAW_FILES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_csvs() -> dict[str, pd.DataFrame]:
    """Read all required raw CSVs from data/ into memory. Fails fast with a clear
    message if a file is missing, since the whole pipeline depends on this data."""
    frames = {}
    missing = []
    for key, filename in RAW_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        logger.info(f"Loading {filename} ...")
        frames[key] = pd.read_csv(path)
        logger.info(f"  -> {filename}: {frames[key].shape[0]:,} rows, {frames[key].shape[1]} cols")

    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) in {DATA_DIR}: {missing}. "
            f"Download them from the Kaggle Home Credit Default Risk competition "
            f"and place them in the data/ folder before running this pipeline."
        )
    return frames


def build_duckdb(frames: dict[str, pd.DataFrame] | None = None, db_path: str = DB_PATH) -> None:
    """Persist raw tables into a DuckDB file. This is what the NL-to-SQL chatbot
    queries directly, so table/column names here should match what we tell the LLM
    about the schema in src/talk_to_data/prompt_templates.py."""
    if frames is None:
        frames = load_raw_csvs()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)

    table_map = {
        "app_train": "applications",
        "bureau": "bureau",
        "bureau_balance": "bureau_balance",
        "previous_application": "previous_applications",
        "pos_cash": "pos_cash_balance",
        "credit_card_balance": "credit_card_balance",
    }

    for key, table_name in table_map.items():
        if key not in frames:
            continue
        df = frames[key]
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
        logger.info(f"Wrote table '{table_name}' ({len(df):,} rows) to {db_path}")

    con.close()
    logger.info(f"DuckDB database ready at {db_path}")


if __name__ == "__main__":
    build_duckdb()
