"""
Shared action definitions for the Self-Driving Car Simulation.

Both agents (R — DQN, S — SARSA) use the same discrete action set
derived from HighwayEnv's DiscreteMetaAction space.

This module is the single source of truth for action indices, names,
and validation.  It knows nothing about agents, training, or Streamlit.
"""

from enum import IntEnum
from typing import Optional


class Action(IntEnum):
    """
    The five high-level tactical actions available to every vehicle.

    These map 1-to-1 to HighwayEnv's ``DiscreteMetaAction`` indices.
    """

    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ACTIONS: int = len(Action)
"""Total number of discrete actions (5)."""

ACTION_NAMES: dict[int, str] = {a.value: a.name for a in Action}
"""Maps action index → human-readable name, e.g. {0: 'LANE_LEFT', …}."""

ACTION_INDEX: dict[str, int] = {a.name: a.value for a in Action}
"""Maps action name → index, e.g. {'LANE_LEFT': 0, …}."""

# Keyboard mapping — used by the human recorder (Build Step 2).
# Keys are lowercase key names from the native PyGame event loop.
KEYBOARD_ACTION_MAP: dict[str, int] = {
    # Arrow keys
    "arrowleft": Action.LANE_LEFT,
    "arrowright": Action.LANE_RIGHT,
    "arrowup": Action.FASTER,
    "arrowdown": Action.SLOWER,
    # WASD
    "a": Action.LANE_LEFT,
    "d": Action.LANE_RIGHT,
    "w": Action.FASTER,
    "s": Action.SLOWER,
    # Spacebar → idle / emergency brake
    " ": Action.IDLE,
    "spacebar": Action.IDLE,
}
"""Maps keyboard key names to action indices for human driving."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_action(action: int) -> bool:
    """Return True if *action* is a valid action index (0–4 inclusive)."""
    return isinstance(action, int) and 0 <= action < NUM_ACTIONS


def validate_action(action: int) -> int:
    """
    Return *action* unchanged if valid; raise ``ValueError`` otherwise.

    Use this at trust boundaries (loading human data, receiving UI input).
    """
    if not is_valid_action(action):
        raise ValueError(
            f"Invalid action {action!r}. "
            f"Expected an integer in 0..{NUM_ACTIONS - 1}."
        )
    return action


def action_name(action: int) -> str:
    """Return the human-readable name for a valid action index."""
    validate_action(action)
    return ACTION_NAMES[action]


def keyboard_to_action(key: str) -> Optional[int]:
    """
    Convert a keyboard key name to an action index.

    Returns ``None`` if the key is not mapped to any action.
    """
    return KEYBOARD_ACTION_MAP.get(key.lower())
