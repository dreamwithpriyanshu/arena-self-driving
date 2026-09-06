"""UI-neutral helpers for loading dashboard data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def history_files(directory: str | Path = "artifacts") -> list[Path]:
    """Return timestamped training-history files newest first."""
    root = Path(directory)
    files = sorted(
        root.glob("training_history_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files


def load_history(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited training records."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_histories(paths: list[Path]) -> list[dict[str, Any]]:
    """Load records from multiple timestamped runs with a stable run label."""
    records: list[dict[str, Any]] = []
    for path in paths:
        for record in load_history(path):
            records.append({"run": path.stem.removeprefix("training_history_"), **record})
    return records


def load_latest_history(directory: str | Path = "artifacts") -> tuple[list[dict[str, Any]], str]:
    """Load the newest available history file and return records plus its path."""
    files = history_files(directory)
    if not files:
        return [], ""
    try:
        return load_history(files[0]), str(files[0])
    except (OSError, json.JSONDecodeError):
        return [], str(files[0])
