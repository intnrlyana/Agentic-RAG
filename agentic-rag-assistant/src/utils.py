"""Shared helpers for the project."""

from pathlib import Path


def ensure_data_dir(path: str = "data") -> Path:
    """Return the data directory path and create it if needed."""
    data_dir = Path(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
