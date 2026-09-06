"""
src/talk_to_data/nl_to_sql.py

The talk-to-data chatbot: natural language question -> LLM generates SQL ->
SQL is validated (src/talk_to_data/query_runner.py) -> executed against DuckDB
-> LLM summarizes the result in plain English.

The LLM NEVER touches the database directly. Its SQL output always passes
through validate_sql()/execute_query() first — this is the hallucination/
injection control layer, and it's enforced in code, not just prompted for.
"""
from groq import Groq

from src.talk_to_data.prompt_templates import build_sql_prompt, build_answer_prompt
from src.talk_to_data.query_runner import execute_query, SQLValidationError
from src.utils.config import GROQ_API_KEY, GROQ_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_sql(question: str) -> str:
    """Ask the LLM to translate the question into SQL. Low temperature for
    consistent, deterministic-leaning SQL generation (token optimization: this
    also reduces retries from malformed/creative SQL)."""
    client = get_client()
    messages = build_sql_prompt(question)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=400,
    )
    sql = response.choices[0].message.content.strip()
    # Strip accidental markdown code fences if the model adds them despite instructions
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def summarize_result(question: str, sql: str, result_df) -> str:
    """Ask the LLM to turn the raw query result into a plain-English answer.
    Grounded strictly in the actual returned rows (passed in full to the
    prompt) to prevent the model from inventing numbers."""
    client = get_client()
    result_rows = result_df.to_dict(orient="records")
    messages = build_answer_prompt(question, sql, result_rows)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def ask(question: str) -> dict:
    """Full pipeline for one chatbot turn. Never raises to the caller — all
    failure modes (bad SQL, validation rejection, execution error, LLM error)
    are caught and returned as a structured, user-safe response."""
    try:
        raw_sql = generate_sql(question)
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {"success": False, "answer": f"I couldn't process that question: {e}", "sql": None, "data": None}

    if raw_sql.upper().startswith("NO_QUERY"):
        reason = raw_sql.split(":", 1)[-1].strip() if ":" in raw_sql else "Not answerable from this dataset."
        return {"success": False, "answer": f"I can't answer that from the credit risk data. {reason}",
                "sql": None, "data": None}

    try:
        result_df = execute_query(raw_sql)
    except SQLValidationError as e:
        logger.warning(f"SQL rejected by validator: {e} | SQL was: {raw_sql}")
        return {"success": False,
                "answer": "I generated a query that didn't pass safety validation, so I won't run it. "
                          "Try rephrasing your question.",
                "sql": raw_sql, "data": None}
    except Exception as e:
        logger.error(f"SQL execution error: {e} | SQL was: {raw_sql}")
        return {"success": False, "answer": "That query failed to run against the database. Try rephrasing.",
                "sql": raw_sql, "data": None}

    try:
        answer_text = summarize_result(question, raw_sql, result_df)
    except Exception as e:
        logger.error(f"Answer summarization failed: {e}")
        answer_text = f"Query ran successfully and returned {len(result_df)} row(s), " \
                       f"but I couldn't generate a summary: {e}"

    return {"success": True, "answer": answer_text, "sql": raw_sql, "data": result_df}


# --- Suggested queries the UI can show as quick-start buttons (satisfies the
# "at least 5 working query patterns" requirement with pre-validated examples) ---
SAMPLE_QUESTIONS = [
    "What is the average income of applicants who defaulted vs those who didn't?",
    "How many applicants have more than 2 overdue bureau loans?",
    "What's the default rate for applicants with a college education?",
    "Show me the top 10 organization types by average credit amount.",
    "How many applicants were previously refused a loan by this lender?",
    "What is the average credit amount for female vs male applicants?",
    "What's the average credit card utilization for applicants who defaulted?",
]
