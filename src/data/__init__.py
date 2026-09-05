"""
src.data — Data recording, validation, and loading.

Handles human demonstrations and autonomous logs. Pure functions
and dataclasses with no agent or training logic.
"""

from src.data.loader import get_dataset_summary, load_all_episodes, load_all_transitions, load_episode
from src.data.recorder import TransitionRecorder
from src.data.schemas import TRANSITION_REQUIRED_FIELDS, EpisodeMetadata, Transition
from src.data.validator import ValidationResult, validate_directory, validate_episode_file

__all__ = [
    "EpisodeMetadata",
    "Transition",
    "TRANSITION_REQUIRED_FIELDS",
    "TransitionRecorder",
    "ValidationResult",
    "validate_episode_file",
    "validate_directory",
    "load_episode",
    "load_all_episodes",
    "load_all_transitions",
    "get_dataset_summary",
]
