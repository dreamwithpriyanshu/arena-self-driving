"""Filesystem locations used by the application and training workers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = PROJECT_ROOT / "published"


def storage_root() -> Path:
    """Return the writable root, preserving the repository layout locally."""
    configured = os.environ.get("ARENA_STORAGE_DIR")
    root = Path(configured).expanduser() if configured else PROJECT_ROOT
    return root.resolve()


def data_dir() -> Path:
    return storage_root() / "data"


def demonstrations_dir() -> Path:
    return data_dir() / "human_demonstrations"


def artifacts_dir() -> Path:
    return storage_root() / "artifacts"


def checkpoints_dir() -> Path:
    return artifacts_dir() / "checkpoints"


def ensure_storage_dirs() -> None:
    demonstrations_dir().mkdir(parents=True, exist_ok=True)
    checkpoints_dir().mkdir(parents=True, exist_ok=True)
    (artifacts_dir() / "runs").mkdir(parents=True, exist_ok=True)
    published_checkpoint = PUBLISHED_DIR / "checkpoints" / "sarsa_q_table.npy"
    target_checkpoint = checkpoints_dir() / "sarsa_q_table.npy"
    if published_checkpoint.exists() and not target_checkpoint.exists():
        shutil.copy2(published_checkpoint, target_checkpoint)
    published_history = PUBLISHED_DIR / "history"
    target_history = artifacts_dir() / "runs" / "published"
    if published_history.is_dir() and not target_history.exists():
        shutil.copytree(published_history, target_history)
