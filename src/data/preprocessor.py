"""
src/data/preprocessor.py

Builds the model-ready feature table by:
  1. Cleaning the main application table (anomalies, missing values)
  2. Aggregating bureau.csv (external credit history) to one row per SK_ID_CURR
  3. Aggregating previous_application.csv (prior loans with this lender)
  4. Aggregating POS_CASH_balance.csv (monthly repayment status / DPD)
  5. Joining everything into a single feature matrix

Aggregations run in DuckDB (SQL) rather than pandas groupby: POS_CASH_balance
alone is 10M+ rows, and pandas groupby with lambda aggregations at that scale is
slow and memory-hungry. DuckDB streams and aggregates it in seconds using far
less RAM, which is exactly the workload it's built for.

This multi-table aggregation is what separates this pipeline from a single-table
baseline: repayment behaviour and external bureau history are consistently among
the strongest predictors of default in this dataset.
"""
import duckdb
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Home Credit encodes "no date" as this sentinel in several DAYS_ columns
DAYS_EMPLOYED_ANOMALY = 365243


def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Known anomaly: DAYS_EMPLOYED has a sentinel value for "not employed"/pensioners
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # Convert "days before application" fields (negative) into positive years for readability
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        if col in df.columns:
            df[col.replace("DAYS_", "YEARS_")] = (-df[col] / 365.25).round(1)

    # Core engineered ratios — strong, interpretable risk signals
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM_YEARS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"] / 12
    df["EMPLOYED_TO_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    return df


def aggregate_bureau(con: duckdb.DuckDBPyConnection, table: str = "bureau") -> pd.DataFrame:
    """One row per SK_ID_CURR summarising external credit bureau history. Computed
    in SQL for speed/memory efficiency on a 1.7M-row table."""
    return con.execute(f"""
        SELECT
            SK_ID_CURR,
            COUNT(SK_ID_BUREAU)                                        AS bureau_loan_count,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)  AS bureau_active_loan_count,
            SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)    AS bureau_overdue_loan_count,
            MAX(CREDIT_DAY_OVERDUE)                                    AS bureau_max_days_overdue,
            SUM(AMT_CREDIT_SUM_DEBT)                                   AS bureau_total_debt,
            SUM(AMT_CREDIT_SUM)                                        AS bureau_total_credit,
            MAX(AMT_CREDIT_MAX_OVERDUE)                                AS bureau_max_overdue_amt,
            SUM(CNT_CREDIT_PROLONG)                                    AS bureau_credit_prolonged_count,
            SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0)  AS bureau_debt_credit_ratio
        FROM {table}
        GROUP BY SK_ID_CURR
    """).fetchdf()


def aggregate_previous_applications(con: duckdb.DuckDBPyConnection, table: str = "previous_applications") -> pd.DataFrame:
    """One row per SK_ID_CURR summarising the applicant's prior loans with this lender."""
    return con.execute(f"""
        SELECT
            SK_ID_CURR,
            COUNT(SK_ID_PREV)                                                  AS prev_application_count,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_approved_count,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)  AS prev_refused_count,
            AVG(AMT_CREDIT)                                                    AS prev_avg_credit_amt,
            AVG(AMT_ANNUITY)                                                   AS prev_avg_annuity,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(SK_ID_PREV), 0)                                 AS prev_refusal_rate
        FROM {table}
        GROUP BY SK_ID_CURR
    """).fetchdf()


def aggregate_pos_cash(con: duckdb.DuckDBPyConnection, table: str = "pos_cash_balance") -> pd.DataFrame:
    """One row per SK_ID_CURR summarising monthly POS/cash loan balance history.
    SK_DPD (days past due) and SK_DPD_DEF (days past due, ignoring tolerance) are
    direct, month-by-month measurements of repayment behaviour — one of the
    strongest available signals of future default risk. Aggregated in SQL since
    this table alone is 10M+ rows."""
    return con.execute(f"""
        SELECT
            SK_ID_CURR,
            COUNT(MONTHS_BALANCE)                                                AS pos_months_count,
            AVG(SK_DPD)                                                          AS pos_avg_dpd,
            MAX(SK_DPD)                                                          AS pos_max_dpd,
            AVG(SK_DPD_DEF)                                                      AS pos_avg_dpd_def,
            MAX(SK_DPD_DEF)                                                      AS pos_max_dpd_def,
            SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)                          AS pos_months_with_dpd,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1 ELSE 0 END)  AS pos_completed_contracts,
            SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(MONTHS_BALANCE), 0)                               AS pos_dpd_rate
        FROM {table}
        GROUP BY SK_ID_CURR
    """).fetchdf()


def aggregate_bureau_balance(con: duckdb.DuckDBPyConnection, table: str = "bureau_balance") -> pd.DataFrame:
    """One row per SK_ID_CURR summarising monthly bureau credit-line status
    history. STATUS is a DPD bucket ('0'=no overdue, '1'-'5'=increasing overdue
    severity, 'C'=closed, 'X'=unknown) — joined through bureau (SK_ID_BUREAU)
    since bureau_balance itself has no SK_ID_CURR. Aggregated in SQL since this
    table is large (27M+ rows in the full dataset)."""
    return con.execute(f"""
        SELECT
            b.SK_ID_CURR,
            COUNT(*)                                                              AS bb_months_count,
            SUM(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1 ELSE 0 END)    AS bb_months_overdue,
            SUM(CASE WHEN bb.STATUS = 'C' THEN 1 ELSE 0 END)                       AS bb_months_closed,
            MAX(CASE WHEN bb.STATUS IN ('1','2','3','4','5')
                     THEN CAST(bb.STATUS AS INTEGER) ELSE 0 END)                   AS bb_max_dpd_bucket,
            SUM(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(*), 0)                                              AS bb_overdue_rate
        FROM {table} bb
        JOIN bureau b ON bb.SK_ID_BUREAU = b.SK_ID_BUREAU
        GROUP BY b.SK_ID_CURR
    """).fetchdf()


def aggregate_credit_card_balance(con: duckdb.DuckDBPyConnection, table: str = "credit_card_balance") -> pd.DataFrame:
    """One row per SK_ID_CURR summarising monthly credit-card balance and
    utilization history. Credit utilization (balance / credit limit) is a
    classic, strong risk signal in consumer credit scoring — high sustained
    utilization predicts default independently of raw balance size."""
    return con.execute(f"""
        SELECT
            SK_ID_CURR,
            COUNT(*)                                                     AS cc_months_count,
            AVG(AMT_BALANCE)                                             AS cc_avg_balance,
            AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))         AS cc_avg_utilization,
            MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))         AS cc_max_utilization,
            AVG(SK_DPD)                                                  AS cc_avg_dpd,
            MAX(SK_DPD)                                                  AS cc_max_dpd,
            SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)                  AS cc_months_with_dpd,
            SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(*), 0)                                    AS cc_dpd_rate
        FROM {table}
        GROUP BY SK_ID_CURR
    """).fetchdf()


def build_feature_table(
    app: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    use_bureau: bool = True,
    use_bureau_balance: bool = True,
    use_previous_application: bool = True,
    use_pos_cash: bool = True,
    use_credit_card_balance: bool = True,
) -> pd.DataFrame:
    """Full pipeline: clean the main table, aggregate the auxiliary tables in
    DuckDB, and left-join everything on SK_ID_CURR in pandas (the aggregated
    tables are small — one row per applicant — so this final join is cheap even
    though the source tables are millions of rows). Applicants with no bureau/
    prior-loan/POS-cash/credit-card history get NaN for those features (handled
    downstream by the model's native missing-value support)."""
    logger.info("Cleaning application table ...")
    df = clean_applications(app)

    if use_bureau:
        logger.info("Aggregating bureau (SQL) ...")
        df = df.merge(aggregate_bureau(con), on="SK_ID_CURR", how="left")

    if use_bureau_balance:
        logger.info("Aggregating bureau_balance (SQL) ...")
        df = df.merge(aggregate_bureau_balance(con), on="SK_ID_CURR", how="left")

    if use_previous_application:
        logger.info("Aggregating previous_applications (SQL) ...")
        df = df.merge(aggregate_previous_applications(con), on="SK_ID_CURR", how="left")

    if use_pos_cash:
        logger.info("Aggregating pos_cash_balance (SQL) ...")
        df = df.merge(aggregate_pos_cash(con), on="SK_ID_CURR", how="left")

    if use_credit_card_balance:
        logger.info("Aggregating credit_card_balance (SQL) ...")
        df = df.merge(aggregate_credit_card_balance(con), on="SK_ID_CURR", how="left")

    logger.info(f"Final feature table shape: {df.shape}")
    return df
