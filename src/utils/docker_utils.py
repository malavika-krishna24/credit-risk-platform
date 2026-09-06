"""
src/utils/docker_utils.py

Small utilities for validating the data/ mount and resolving paths that need
to work identically whether the app is running locally or inside the Docker
container (where data/ and models/ are mounted volumes — see docker-compose.yml).
"""
from pathlib import Path

from src.utils.config import DATA_DIR, RAW_FILES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_required_files(raw_files: dict[str, str] | None = None, data_dir: Path | None = None) -> dict[str, bool]:
    """Check which of the required raw CSVs are present in data/, without
    raising. Used by loader.py to build a clear, actionable error message,
    and can be called standalone (e.g. as a Docker healthcheck / entrypoint
    sanity check) to fail fast with a helpful message before the app starts
    rather than partway through training."""
    raw_files = raw_files or RAW_FILES
    data_dir = data_dir or DATA_DIR

    status = {}
    for key, filename in raw_files.items():
        status[filename] = (data_dir / filename).exists()
    return status


def missing_files_message(raw_files: dict[str, str] | None = None, data_dir: Path | None = None) -> str | None:
    """Returns a human-readable message listing missing files, or None if
    everything required is present. Kept separate from check_required_files
    so callers can use the raw status dict OR just this one-liner, whichever
    fits (loader.py uses the message form)."""
    status = check_required_files(raw_files, data_dir)
    missing = [f for f, present in status.items() if not present]
    if not missing:
        return None
    return (
        f"Missing required file(s) in {data_dir or DATA_DIR}: {missing}. "
        f"Download them from the Kaggle Home Credit Default Risk competition "
        f"and place them in the data/ folder (or the mounted volume, if running "
        f"in Docker) before running this pipeline."
    )


def is_running_in_docker() -> bool:
    """Best-effort check for whether the current process is inside a Docker
    container — useful for a couple of UI/logging messages that differ
    slightly (e.g. pointing to a mounted-volume path vs. a local relative
    path) without needing a separate env var for it."""
    return Path("/.dockerenv").exists()
