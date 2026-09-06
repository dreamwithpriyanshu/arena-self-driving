"""Publish the current local SARSA baseline for GitHub installations."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import PROJECT_ROOT


def main() -> None:
    checkpoint = PROJECT_ROOT / "artifacts" / "checkpoints" / "sarsa_q_table.npy"
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    published = PROJECT_ROOT / "published"
    published_checkpoint = published / "checkpoints" / "sarsa_q_table.npy"
    published_history = published / "history"
    published_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    published_history.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, published_checkpoint)

    histories = sorted((PROJECT_ROOT / "artifacts").glob("training_history_*.jsonl"))
    if histories:
        history = histories[-1]
        metadata = history.with_suffix(".meta.json")
        shutil.copy2(history, published_history / history.name)
        if metadata.exists():
            shutil.copy2(metadata, published_history / metadata.name)
    print("Published the SARSA checkpoint and latest training history.")
    print("Review the published/ files, then commit and push them to GitHub.")


if __name__ == "__main__":
    main()
