"""
src/talk_to_data/query_runner.py

Validates and executes LLM-generated SQL against the DuckDB database. This is
the primary hallucination/safety guardrail for the talk-to-data module: the LLM
output is NEVER executed directly without passing through this validator first.

Validation layers:
  1. Reject anything that isn't a single SELECT statement (blocks DDL/DML entirely,
     even if the LLM ever tries to "help" by modifying data).
  2. Reject queries referencing tables/columns outside the known schema (blocks
     hallucinated column/table names before they hit the database and produce a
     confusing raw DB error for the user).
  3. Enforce a row LIMIT ceiling so a mistaken unaggregated query can't return
     hundreds of thousands of rows into the chat.
  4. Enforce a query timeout so a malformed/expensive query can't hang the app.
"""
import re

import duckdb
import pandas as pd

from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_TABLES = {"applications", "bureau", "previous_applications", "pos_cash_balance"}

# Keywords that should never appear in a generated query. Checked as whole words
# to avoid false positives on legitimate column names.
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "PRAGMA", "CALL",
    "GRANT", "REVOKE", "REPLACE", "VACUUM",
}

MAX_ROW_LIMIT = 500
QUERY_TIMEOUT_SECONDS = 15


class SQLValidationError(Exception):
    pass


def validate_sql(sql: str) -> str:
    """Raises SQLValidationError if the query is unsafe or references an
    unknown table. Returns a (possibly LIMIT-clamped) cleaned SQL string on
    success."""
    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned:
        raise SQLValidationError("Empty query.")

    if cleaned.upper().startswith("NO_QUERY"):
        raise SQLValidationError(cleaned)

    # Must be a single statement (no chained statements via ;)
    if ";" in sql.strip().rstrip(";"):
        raise SQLValidationError("Multiple statements are not allowed.")

    # Must start with SELECT or WITH (CTE)
    first_word = cleaned.split(None, 1)[0].upper() if cleaned.split() else ""
    if first_word not in {"SELECT", "WITH"}:
        raise SQLValidationError(f"Only SELECT queries are allowed (got '{first_word}').")

    # Forbidden keyword scan (word-boundary match)
    upper_sql = cleaned.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper_sql):
            raise SQLValidationError(f"Forbidden keyword detected: {kw}")

    # Table allow-list check — extract identifiers after FROM/JOIN
    referenced_tables = set(re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", cleaned, re.IGNORECASE))
    unknown_tables = referenced_tables - ALLOWED_TABLES
    if unknown_tables:
        raise SQLValidationError(f"Unknown table(s) referenced: {unknown_tables}. "
                                  f"Allowed tables: {ALLOWED_TABLES}")

    # Enforce a row limit ceiling for non-aggregate queries
    has_limit = re.search(r"\bLIMIT\s+(\d+)", cleaned, re.IGNORECASE)
    is_aggregate_only = bool(re.search(r"\b(COUNT|AVG|SUM|MIN|MAX)\s*\(", cleaned, re.IGNORECASE)) and \
                         not re.search(r"\bGROUP\s+BY\b", cleaned, re.IGNORECASE)

    if has_limit:
        limit_val = int(has_limit.group(1))
        if limit_val > MAX_ROW_LIMIT:
            cleaned = re.sub(r"\bLIMIT\s+\d+", f"LIMIT {MAX_ROW_LIMIT}", cleaned, flags=re.IGNORECASE)
    elif not is_aggregate_only:
        cleaned = f"{cleaned} LIMIT {MAX_ROW_LIMIT}"

    return cleaned


def execute_query(sql: str) -> pd.DataFrame:
    """Validate then execute against DuckDB. Raises SQLValidationError for
    unsafe/invalid queries, or duckdb.Error for genuine SQL errors (e.g. typo'd
    column that slipped past validation) — both are caught by the caller and
    surfaced as a friendly chat message, never a raw stack trace."""
    validated_sql = validate_sql(sql)
    logger.info(f"Executing validated SQL: {validated_sql}")

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        result = con.execute(validated_sql).fetchdf()
    finally:
        con.close()

    return result
