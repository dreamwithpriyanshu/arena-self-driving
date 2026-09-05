"""
Data schemas — shared data structures for transitions and episodes.

These dataclasses define the canonical format for human demonstrations
and autonomous logs.  Every module that records, loads, or validates
data uses these types.

Layer: data  (pure data definitions, no side effects)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Transition:
    """
    A single (s, a, r, s', done) transition recorded during driving.

    This is the atomic unit of training data for both DQN and SARSA.
    """

    step: int
    """Step index within the episode (0-based)."""

    state: list[float]
    """Flat continuous state vector (for DQN).  Length = V * F."""

    action: int
    """Discrete action index (0–4)."""

    reward: float
    """Scalar reward from the environment."""

    next_state: list[float]
    """State vector after taking the action."""

    terminated: bool
    """True if the episode ended (e.g., collision)."""

    truncated: bool
    """True if the episode was truncated by Gymnasium."""

    discrete_state: int
    """Integer state index for tabular SARSA."""

    next_discrete_state: int
    """Discrete state index after taking the action."""

    lane: int = 0
    """Ego vehicle's lane index at this step."""

    speed: float = 0.0
    """Ego vehicle's speed (m/s) at this step."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the transition was recorded."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        """
        Construct a ``Transition`` from a dict (e.g., parsed JSON line).

        Unknown keys are silently ignored so that older/newer file
        formats remain forward-/backward-compatible.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class EpisodeMetadata:
    """
    Metadata header for a recorded episode file.

    Written as the first line of each ``.jsonl`` file (with
    ``"_type": "episode_metadata"``).
    """

    episode_id: str
    """Unique identifier (e.g., ``'ep_20260906_012345_001'``)."""

    vehicle: str = "R"
    """Which vehicle was being driven: ``'R'`` (DQN) or ``'S'`` (SARSA)."""

    source: str = "human"
    """Data source: ``'human'`` or ``'autonomous'``."""

    num_transitions: int = 0
    """Number of transitions in the episode (filled on save)."""

    total_reward: float = 0.0
    """Cumulative reward for the episode (filled on save)."""

    start_time: float = field(default_factory=time.time)
    """Unix timestamp when the episode started."""

    end_time: float = 0.0
    """Unix timestamp when the episode ended (filled on save)."""

    seed: Optional[int] = None
    """Environment seed, if set."""

    config_overrides: dict[str, Any] = field(default_factory=dict)
    """Any env config overrides used for this episode."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict with a ``_type`` discriminator."""
        d = asdict(self)
        d["_type"] = "episode_metadata"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeMetadata":
        """Construct from a dict, ignoring the ``_type`` key."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items()
                    if k in known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Required fields for validation
# ---------------------------------------------------------------------------

TRANSITION_REQUIRED_FIELDS: set[str] = {
    "step", "state", "action", "reward", "next_state",
    "terminated", "truncated", "discrete_state", "next_discrete_state",
}
"""The minimum set of keys a transition dict must contain to be valid."""
