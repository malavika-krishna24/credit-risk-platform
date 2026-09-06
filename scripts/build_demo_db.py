"""
scripts/build_demo_db.py

Builds a size-reduced DuckDB database for the public cloud deployment
(Streamlit Community Cloud). The full credit_risk.duckdb (all 6 tables,
27M+ rows in bureau_balance alone) is several hundred MB — too large to
commit to a git repo (GitHub's hard limit is 100MB per file).

This script instead samples N_SAMPLE_APPLICANTS real applicants (default
30,000) and filters every table down to only rows for those applicants,
preserving real, correct relationships between tables. This is NOT fake or
synthetic data — every row is a real applicant from the actual dataset, just
a representative subset.

The full-size Overview/EDA/Explainability/Business Rules numbers shown in
the app are unaffected by this: those come from `models/model_metadata.json`
and `documents/derived_business_rules.csv`, which are already committed and
were computed from the FULL 307,511-applicant dataset during real training.
Only the Talk-to-Data chatbot's underlying query results differ slightly on
the deployed demo vs. a full local/Docker setup — clearly documented in the
README and in the app itself.

Usage:
    python scripts/build_demo_db.py
"""
import sys
sys.path.insert(0, ".")

import duckdb
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_SAMPLE_APPLICANTS = 30000
DEMO_DB_PATH = "data/credit_risk_demo.duckdb"


def main():
    import os

    # ATTACH lets us query the source DB directly in SQL, without ever pulling
    # a full table into pandas memory first — the earlier pandas-based version
    # OOM'd on previous_applications (1.7M rows) because it materialized the
    # entire table before filtering, when only ~30K rows were ever needed.
    dst = duckdb.connect(DEMO_DB_PATH)
    dst.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")

    existing_tables = {row[0] for row in dst.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src'").fetchall()}
    logger.info(f"Source tables found: {existing_tables}")

    if "applications" not in existing_tables:
        raise RuntimeError("Source DB has no 'applications' table — run src/data/loader.py first.")

    total_applicants = dst.execute("SELECT COUNT(*) FROM src.applications").fetchone()[0]
    sample_size = min(N_SAMPLE_APPLICANTS, total_applicants)
    logger.info(f"Sampling {sample_size:,} of {total_applicants:,} applicants for the demo DB ...")

    dst.execute(f"""
        CREATE OR REPLACE TABLE _sampled_ids AS
        SELECT SK_ID_CURR FROM src.applications USING SAMPLE {sample_size} ROWS
    """)

    table_join_cols = {
        "applications": "SK_ID_CURR",
        "bureau": "SK_ID_CURR",
        "previous_applications": "SK_ID_CURR",
        "pos_cash_balance": "SK_ID_CURR",
        "credit_card_balance": "SK_ID_CURR",
    }

    for table, join_col in table_join_cols.items():
        if table not in existing_tables:
            logger.warning(f"Skipping '{table}' — not present in source DB.")
            continue
        dst.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT t.* FROM src.{table} t
            JOIN _sampled_ids s ON t.{join_col} = s.SK_ID_CURR
        """)
        row_count = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table}: {row_count:,} rows in demo DB")

    # bureau_balance has no SK_ID_CURR — must go through bureau (SK_ID_BUREAU)
    if "bureau_balance" in existing_tables and "bureau" in existing_tables:
        dst.execute("""
            CREATE OR REPLACE TABLE bureau_balance AS
            SELECT bb.* FROM src.bureau_balance bb
            JOIN bureau b ON bb.SK_ID_BUREAU = b.SK_ID_BUREAU
        """)
        row_count = dst.execute("SELECT COUNT(*) FROM bureau_balance").fetchone()[0]
        logger.info(f"  bureau_balance: {row_count:,} rows in demo DB")

    dst.execute("DROP TABLE _sampled_ids")
    dst.execute("DETACH src")
    dst.close()

    size_mb = os.path.getsize(DEMO_DB_PATH) / (1024 * 1024)
    logger.info(f"Demo DB written to {DEMO_DB_PATH} ({size_mb:.1f} MB)")
    if size_mb > 90:
        logger.warning(f"Demo DB is {size_mb:.1f}MB — close to GitHub's 100MB limit. "
                        f"Consider lowering N_SAMPLE_APPLICANTS.")


if __name__ == "__main__":
    main()
