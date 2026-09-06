"""
src/utils/helpers.py

Small, generic utility functions shared across the data, ML, and UI layers.
Kept separate from config.py (settings/constants) and logger.py (logging setup)
so each utils file has one clear responsibility.
"""
from pathlib import Path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of raising on a zero
    denominator — used throughout preprocessor.py's ratio features
    (e.g. debt/credit, DPD rate) where a zero denominator is a real,
    expected case (an applicant with no bureau credit at all), not an error."""
    if denominator in (0, 0.0) or denominator is None:
        return default
    return numerator / denominator


def format_currency(amount: float, currency: str = "$") -> str:
    """Format a raw number as a human-readable currency string for display
    in the UI and logs, e.g. 1234567.8 -> '$1,234,568'."""
    if amount is None:
        return "N/A"
    return f"{currency}{amount:,.0f}"


def format_percent(fraction: float, decimals: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string, e.g. 0.0807 -> '8.1%'."""
    if fraction is None:
        return "N/A"
    return f"{fraction * 100:.{decimals}f}%"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate long text (e.g. an LLM answer or SQL query) for compact
    logging, appending an ellipsis if truncation occurred."""
    if text is None:
        return ""
    return text if len(text) <= max_length else text[:max_length].rstrip() + "..."


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist, and
    return it as a Path — a small convenience used by scripts that write
    output files (charts, artifacts) to a folder that may not exist yet."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
