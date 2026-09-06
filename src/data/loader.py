"""
Data loader — reads validated JSONL episode files into memory.

Provides functions to load individual episodes or entire directories
of human demonstrations / autonomous logs.  Always validates before
returning data.

Layer: data  (depends on schemas, validator; knows nothing about agents/UI)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.data.schemas import EpisodeMetadata, Transition
from src.data.validator import validate_episode_file
from src.storage import demonstrations_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (for resolving relative paths)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Single-episode loader
# ---------------------------------------------------------------------------

def load_episode(
    filepath: str | Path,
    validate: bool = True,
    expected_state_dim: Optional[int] = None,
) -> tuple[EpisodeMetadata, list[Transition]]:
    """
    Load a single JSONL episode file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.jsonl`` file.
    validate : bool
        If True, validates the file before loading.  Raises ValueError
        if validation fails.
    expected_state_dim : int, optional
        Expected state vector length (passed to validator).

    Returns
    -------
    tuple[EpisodeMetadata, list[Transition]]
        The episode metadata and list of transitions.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If validation fails (and ``validate=True``).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Episode file not found: {filepath}")

    if validate:
        result = validate_episode_file(filepath, expected_state_dim)
        if not result.is_valid:
            error_summary = "; ".join(result.errors[:5])
            raise ValueError(
                f"Validation failed for {filepath}: {error_summary}"
            )

    metadata = None
    transitions: list[Transition] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            if line_num == 1 and raw.get("_type") == "episode_metadata":
                metadata = EpisodeMetadata.from_dict(raw)
            else:
                transitions.append(Transition.from_dict(raw))

    if metadata is None:
        # File has no metadata header — create a minimal one
        metadata = EpisodeMetadata(
            episode_id=filepath.stem,
            num_transitions=len(transitions),
        )

    logger.info(
        "Loaded episode %s: %d transitions",
        metadata.episode_id, len(transitions),
    )
    return metadata, transitions


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------

def load_all_episodes(
    directory: str | Path = demonstrations_dir(),
    validate: bool = True,
    expected_state_dim: Optional[int] = None,
    skip_invalid: bool = True,
) -> list[tuple[EpisodeMetadata, list[Transition]]]:
    """
    Load all ``.jsonl`` episode files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing ``.jsonl`` files (relative to project root
        or absolute).
    validate : bool
        Validate each file before loading.
    expected_state_dim : int, optional
        Expected state vector length.
    skip_invalid : bool
        If True, skip files that fail validation instead of raising.

    Returns
    -------
    list[tuple[EpisodeMetadata, list[Transition]]]
        A list of (metadata, transitions) pairs.
    """
    directory = Path(directory)
    if not directory.is_absolute():
        directory = _PROJECT_ROOT / directory

    if not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return []

    episodes = []
    for fpath in sorted(directory.glob("*.jsonl")):
        try:
            ep = load_episode(fpath, validate=validate,
                              expected_state_dim=expected_state_dim)
            episodes.append(ep)
        except (ValueError, FileNotFoundError) as exc:
            if skip_invalid:
                logger.warning("Skipping %s: %s", fpath.name, exc)
            else:
                raise

    logger.info(
        "Loaded %d episodes from %s",
        len(episodes), directory,
    )
    return episodes


# ---------------------------------------------------------------------------
# Convenience: flatten all transitions
# ---------------------------------------------------------------------------

def load_all_transitions(
    directory: str | Path = demonstrations_dir(),
    validate: bool = True,
    expected_state_dim: Optional[int] = None,
    skip_invalid: bool = True,
) -> list[Transition]:
    """
    Load and flatten all transitions from all episodes in a directory.

    Useful for building a unified replay buffer or training dataset.
    """
    episodes = load_all_episodes(
        directory, validate=validate,
        expected_state_dim=expected_state_dim,
        skip_invalid=skip_invalid,
    )
    all_transitions: list[Transition] = []
    for _meta, transitions in episodes:
        all_transitions.extend(transitions)

    logger.info("Total transitions loaded: %d", len(all_transitions))
    return all_transitions


def get_dataset_summary(
    directory: str | Path = demonstrations_dir(),
) -> dict:
    """
    Return a summary of the dataset in a directory.

    Returns a dict with:
    - ``num_episodes``: number of episode files
    - ``total_transitions``: total transitions across all episodes
    - ``total_reward``: sum of all episode rewards
    - ``vehicles``: breakdown by model label (SARSA ``S`` only)
    - ``episodes``: list of episode metadata dicts
    """
    directory = Path(directory)
    if not directory.is_absolute():
        directory = _PROJECT_ROOT / directory

    if not directory.is_dir():
        return {
            "num_episodes": 0,
            "total_transitions": 0,
            "total_reward": 0.0,
            "vehicles": {"S": 0},
            "episodes": [],
        }

    episode_metas = []
    total_transitions = 0
    total_reward = 0.0
    vehicles = {"S": 0}

    for fpath in sorted(directory.glob("*.jsonl")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    raw = json.loads(first_line)
                    if raw.get("_type") == "episode_metadata":
                        meta = EpisodeMetadata.from_dict(raw)
                        episode_metas.append(meta.to_dict())
                        total_transitions += meta.num_transitions
                        total_reward += meta.total_reward
                        if meta.vehicle in vehicles:
                            vehicles[meta.vehicle] += 1
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cannot read metadata from %s: %s", fpath.name, exc)

    return {
        "num_episodes": len(episode_metas),
        "total_transitions": total_transitions,
        "total_reward": total_reward,
        "vehicles": vehicles,
        "episodes": episode_metas,
    }
