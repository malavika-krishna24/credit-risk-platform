-- sql/schema.sql
--
-- Reference schema for the DuckDB database at data/credit_risk.duckdb,
-- created by src/data/loader.py from the raw Kaggle CSVs.
--
-- This file is documentation, not an executable migration — DuckDB tables
-- are actually created dynamically from pandas DataFrames in loader.py
-- (CREATE OR REPLACE TABLE ... AS SELECT * FROM tmp_df), which infers types
-- from the source CSVs. The DDL below describes the resulting structure for
-- reference, and is also what src/talk_to_data/prompt_templates.py describes
-- to the LLM and what query_runner.py's ALLOWED_TABLES enforces.

-- One row per loan applicant. Primary table; SK_ID_CURR is the join key
-- used by every other table in this schema.
CREATE TABLE applications (
    SK_ID_CURR              INTEGER PRIMARY KEY,
    TARGET                  INTEGER,        -- 1 = defaulted, 0 = repaid
    NAME_CONTRACT_TYPE      VARCHAR,
    CODE_GENDER              VARCHAR,
    FLAG_OWN_CAR             VARCHAR,
    FLAG_OWN_REALTY          VARCHAR,
    CNT_CHILDREN             INTEGER,
    AMT_INCOME_TOTAL         DOUBLE,
    AMT_CREDIT               DOUBLE,
    AMT_ANNUITY              DOUBLE,
    NAME_INCOME_TYPE         VARCHAR,
    NAME_EDUCATION_TYPE      VARCHAR,
    NAME_FAMILY_STATUS       VARCHAR,
    NAME_HOUSING_TYPE        VARCHAR,
    DAYS_BIRTH                INTEGER,       -- negative; days before application
    DAYS_EMPLOYED             INTEGER,       -- negative; days before application
    OCCUPATION_TYPE            VARCHAR,
    ORGANIZATION_TYPE          VARCHAR,
    EXT_SOURCE_1                DOUBLE,       -- external credit bureau score, 0-1
    EXT_SOURCE_2                DOUBLE,
    EXT_SOURCE_3                DOUBLE
    -- ...plus ~100 further raw application columns, unchanged from Kaggle's
    -- application_train.csv. Full list: data/HomeCredit_columns_description.csv
);

-- One row per external credit bureau record. Many rows per SK_ID_CURR.
-- Aggregated to one row per applicant in src/data/preprocessor.py::aggregate_bureau().
CREATE TABLE bureau (
    SK_ID_BUREAU           INTEGER PRIMARY KEY,
    SK_ID_CURR              INTEGER REFERENCES applications(SK_ID_CURR),
    CREDIT_ACTIVE            VARCHAR,        -- 'Active' | 'Closed' | 'Sold' | 'Bad debt'
    CREDIT_DAY_OVERDUE        INTEGER,        -- days overdue, 0 if current
    AMT_CREDIT_SUM             DOUBLE,        -- total credit amount
    AMT_CREDIT_SUM_DEBT         DOUBLE,        -- current outstanding debt
    AMT_CREDIT_MAX_OVERDUE       DOUBLE,
    CNT_CREDIT_PROLONG            INTEGER,
    CREDIT_TYPE                    VARCHAR
);

-- One row per prior application this applicant made with THIS lender
-- (not external bureau history — see `bureau` for that).
-- Aggregated in src/data/preprocessor.py::aggregate_previous_applications().
CREATE TABLE previous_applications (
    SK_ID_PREV              INTEGER PRIMARY KEY,
    SK_ID_CURR                INTEGER REFERENCES applications(SK_ID_CURR),
    NAME_CONTRACT_STATUS       VARCHAR,       -- 'Approved' | 'Refused' | 'Cancelled' | 'Unused offer'
    AMT_CREDIT                   DOUBLE,
    AMT_ANNUITY                   DOUBLE,
    DAYS_DECISION                  INTEGER
);

-- One row per applicant per month of an active POS/cash loan.
-- SK_DPD (days past due) is a direct, month-by-month repayment signal.
-- Aggregated in src/data/preprocessor.py::aggregate_pos_cash().
CREATE TABLE pos_cash_balance (
    SK_ID_PREV              INTEGER,
    SK_ID_CURR                INTEGER REFERENCES applications(SK_ID_CURR),
    MONTHS_BALANCE              INTEGER,
    SK_DPD                        INTEGER,     -- days past due that month
    SK_DPD_DEF                     INTEGER,
    NAME_CONTRACT_STATUS            VARCHAR,
    PRIMARY KEY (SK_ID_PREV, MONTHS_BALANCE)
);

-- One row per external bureau credit line per month. NO SK_ID_CURR directly —
-- must join through `bureau` on SK_ID_BUREAU to reach an applicant.
-- STATUS is a DPD bucket: '0' none, '1'-'5' increasing overdue severity,
-- 'C' closed, 'X' unknown. Aggregated in
-- src/data/preprocessor.py::aggregate_bureau_balance().
CREATE TABLE bureau_balance (
    SK_ID_BUREAU            INTEGER REFERENCES bureau(SK_ID_BUREAU),
    MONTHS_BALANCE            INTEGER,
    STATUS                      VARCHAR,
    PRIMARY KEY (SK_ID_BUREAU, MONTHS_BALANCE)
);

-- One row per applicant per month of a credit card. Balance-to-limit
-- (utilization) is a classic, strong consumer-credit risk signal.
-- Aggregated in src/data/preprocessor.py::aggregate_credit_card_balance().
CREATE TABLE credit_card_balance (
    SK_ID_PREV               INTEGER,
    SK_ID_CURR                 INTEGER REFERENCES applications(SK_ID_CURR),
    MONTHS_BALANCE               INTEGER,
    AMT_BALANCE                    DOUBLE,
    AMT_CREDIT_LIMIT_ACTUAL          DOUBLE,
    SK_DPD                            INTEGER,
    NAME_CONTRACT_STATUS                VARCHAR,
    PRIMARY KEY (SK_ID_PREV, MONTHS_BALANCE)
);

-- Note: installments_payments.csv and sample_submission.csv from the full
-- Kaggle download are deliberately not loaded — see README.md section 4 for
-- the reasoning (repayment-timing signal is already covered by
-- pos_cash_balance and credit_card_balance's own DPD tracking).
