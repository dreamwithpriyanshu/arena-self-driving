"""
Transition recorder — writes driving transitions to JSONL files.

Each episode is stored as a single ``.jsonl`` file under
``data/human_demonstrations/`` (for human driving) or
``data/autonomous_logs/`` (for agent evaluation).

File format:
  - Line 1: episode metadata (``{"_type": "episode_metadata", ...}``)
  - Lines 2+: one JSON object per transition

All file writes are flushed after every transition to prevent data loss
on unexpected termination.

Layer: data  (depends on schemas only; knows nothing about envs/agents/UI)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from src.data.schemas import EpisodeMetadata, Transition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allowed base directories for writing (relative to project root)
_ALLOWED_WRITE_DIRS = {
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "artifacts",
}


def _is_safe_path(path: Path) -> bool:
    """
    Return True if *path* is inside one of the allowed write directories.

    Prevents path-traversal attacks when episode names come from user input.
    """
    resolved = path.resolve()
    return any(
        resolved == allowed or allowed in resolved.parents
        for allowed in _ALLOWED_WRITE_DIRS
    )


def _sanitise_filename(name: str) -> str:
    """
    Sanitise a user-provided name for use in filenames.

    Keeps only alphanumeric characters, hyphens, and underscores.
    """
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    # Collapse multiple underscores and strip leading/trailing
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "unnamed"


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TransitionRecorder:
    """
    Buffers and writes transitions to a JSONL episode file.

    Usage::

        rec = TransitionRecorder(base_dir="data/human_demonstrations")
        rec.start_episode(vehicle="R", seed=42)
        rec.record(transition)
        ...
        rec.save_episode()    # finalises and closes the file
        # -or-
        rec.discard_episode() # deletes the file

    Only one episode can be active at a time.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/human_demonstrations",
        source: str = "human",
    ) -> None:
        self._base_dir = (_PROJECT_ROOT / base_dir).resolve()
        self._source = source
        self._file = None
        self._metadata: Optional[EpisodeMetadata] = None
        self._filepath: Optional[Path] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0

        # Validate the base directory
        if not _is_safe_path(self._base_dir):
            raise ValueError(
                f"Base directory {self._base_dir} is outside the allowed "
                f"write directories: {_ALLOWED_WRITE_DIRS}"
            )

        # Ensure directory exists
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TransitionRecorder initialised (base_dir=%s)", self._base_dir)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(
        self,
        vehicle: str = "S",
        seed: Optional[int] = None,
        config_overrides: Optional[dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """
        Begin recording a new episode.

        Parameters
        ----------
        vehicle : str
            ``'R'`` or ``'S'``.
        seed : int, optional
            Environment seed for this episode.
        config_overrides : dict, optional
            Environment config overrides used.
        episode_id : str, optional
            Custom episode ID.  Auto-generated if ``None``.

        Returns
        -------
        str
            The episode ID.

        Raises
        ------
        RuntimeError
            If an episode is already in progress.
        """
        if self._file is not None:
            raise RuntimeError(
                "An episode is already in progress. "
                "Call save_episode() or discard_episode() first."
            )

        if vehicle != "S":
            raise ValueError(f"vehicle must be 'S', got {vehicle!r}")

        if episode_id is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            episode_id = f"ep_{ts}_{vehicle}"

        safe_id = _sanitise_filename(episode_id)
        self._filepath = self._base_dir / f"{safe_id}.jsonl"

        # Double-check safety after building the full path
        if not _is_safe_path(self._filepath):
            raise ValueError(
                f"Computed path {self._filepath} escapes allowed directories."
            )

        self._metadata = EpisodeMetadata(
            episode_id=safe_id,
            vehicle=vehicle,
            source=self._source,
            seed=seed,
            config_overrides=config_overrides or {},
        )

        self._step_count = 0
        self._total_reward = 0.0

        # Open file and write metadata header
        self._file = open(self._filepath, "w", encoding="utf-8")
        self._file.write(json.dumps(self._metadata.to_dict()) + "\n")
        self._file.flush()

        logger.info("Episode started: %s (vehicle=%s)", safe_id, vehicle)
        return safe_id

    def record(self, transition: Transition) -> None:
        """
        Append a single transition to the current episode file.

        Raises
        ------
        RuntimeError
            If no episode is in progress.
        """
        if self._file is None:
            raise RuntimeError("No episode in progress. Call start_episode() first.")

        self._file.write(json.dumps(transition.to_dict()) + "\n")
        self._file.flush()
        self._step_count += 1
        self._total_reward += transition.reward

    def save_episode(self) -> Path:
        """
        Finalise the current episode, update metadata, and close the file.

        Returns
        -------
        Path
            The path to the saved JSONL file.

        Raises
        ------
        RuntimeError
            If no episode is in progress.
        """
        if self._file is None or self._metadata is None or self._filepath is None:
            raise RuntimeError("No episode in progress.")

        # Update metadata with final stats
        self._metadata.num_transitions = self._step_count
        self._metadata.total_reward = self._total_reward
        self._metadata.end_time = time.time()

        # Close the current file
        self._file.close()
        self._file = None

        # Re-write the file with updated metadata on line 1
        self._rewrite_metadata()

        filepath = self._filepath
        logger.info(
            "Episode saved: %s (%d transitions, reward=%.2f)",
            self._metadata.episode_id,
            self._step_count,
            self._total_reward,
        )

        self._metadata = None
        self._filepath = None
        self._step_count = 0
        self._total_reward = 0.0

        return filepath

    def discard_episode(self) -> None:
        """
        Discard the current episode — close and delete the file.
        """
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.exception("Error closing episode file during discard.")
            self._file = None

        if self._filepath is not None and self._filepath.exists():
            try:
                self._filepath.unlink()
                logger.info("Episode discarded: %s", self._filepath.name)
            except OSError:
                logger.exception("Failed to delete discarded episode file.")

        self._metadata = None
        self._filepath = None
        self._step_count = 0
        self._total_reward = 0.0

    @property
    def is_recording(self) -> bool:
        """True if an episode is currently being recorded."""
        return self._file is not None

    @property
    def current_episode_id(self) -> Optional[str]:
        """The ID of the episode currently being recorded, or None."""
        return self._metadata.episode_id if self._metadata else None

    @property
    def current_step_count(self) -> int:
        """Number of transitions recorded in the current episode."""
        return self._step_count

    @property
    def current_total_reward(self) -> float:
        """Cumulative reward in the current episode."""
        return self._total_reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rewrite_metadata(self) -> None:
        """Overwrite line 1 of the file with updated metadata."""
        if self._filepath is None or self._metadata is None:
            return

        lines = []
        with open(self._filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if lines:
            lines[0] = json.dumps(self._metadata.to_dict()) + "\n"

        with open(self._filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def __del__(self) -> None:
        """Clean up open file handle if the recorder is garbage-collected."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
