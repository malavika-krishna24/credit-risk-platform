"""
scripts/build_features.py

Builds data/features_train.parquet from the DuckDB database created by
src/data/loader.py. Run this after `python -m src.data.loader` and before
`python -m src.ml.train`.

Usage:
    python scripts/build_features.py
"""
import sys
sys.path.insert(0, ".")

import duckdb
from src.data.preprocessor import build_feature_table
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    app_train = con.execute("SELECT * FROM applications").fetchdf()
    logger.info(f"Loaded applications table: {app_train.shape}")

    features = build_feature_table(app_train, con)
    features.to_parquet("data/features_train.parquet")
    logger.info(f"Saved data/features_train.parquet: {features.shape}")

    con.close()
