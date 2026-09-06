"""
scripts/test_chatbot_live.py

Run this after setting GROQ_API_KEY in your .env to confirm the talk-to-data
chatbot works end-to-end with a real LLM call. Not part of the automated
pipeline — a manual sanity check for local/deployed environments.

Usage:
    python scripts/test_chatbot_live.py
"""
import sys
sys.path.insert(0, ".")

from src.talk_to_data.nl_to_sql import ask, SAMPLE_QUESTIONS

if __name__ == "__main__":
    print("Running live chatbot test against Groq API...\n")
    for question in SAMPLE_QUESTIONS:
        print("=" * 70)
        print(f"Q: {question}")
        result = ask(question)
        print(f"SQL: {result['sql']}")
        print(f"Success: {result['success']}")
        print(f"Answer: {result['answer']}")
        print()
