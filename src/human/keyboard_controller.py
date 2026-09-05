"""
Keyboard controller — maps user key presses to environment actions.

This module defines the mapping logic only.  The actual key capture
happens in the Streamlit Human Training page (Build Step 4).

Layer: human  (depends on envs.actions; knows nothing about data/training/UI)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.envs.actions import (
    Action,
    KEYBOARD_ACTION_MAP,
    NUM_ACTIONS,
    action_name,
    is_valid_action,
)

logger = logging.getLogger(__name__)


class KeyboardController:
    """
    Translates keyboard key events into discrete environment actions.

    The default mapping uses arrow keys + spacebar::

        Left Arrow  → LANE_LEFT  (0)
        Right Arrow → LANE_RIGHT (2)
        Up Arrow    → FASTER     (3)
        Down Arrow  → SLOWER     (4)
        Space       → IDLE       (1)

    If no key is pressed (or an unmapped key), returns ``IDLE`` by default.

    Usage::

        ctrl = KeyboardController()
        action = ctrl.key_to_action("arrowup")   # → 3 (FASTER)
        action = ctrl.key_to_action("x")          # → 1 (IDLE, default)
    """

    def __init__(
        self,
        default_action: int = Action.IDLE,
        custom_map: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        default_action : int
            Action returned when no valid key mapping is found.
        custom_map : dict, optional
            Override the default key→action mapping.
        """
        if not is_valid_action(default_action):
            raise ValueError(
                f"default_action must be 0–{NUM_ACTIONS-1}, "
                f"got {default_action}"
            )

        self._default_action = default_action
        self._key_map: dict[str, int] = dict(KEYBOARD_ACTION_MAP)

        if custom_map:
            for key, act in custom_map.items():
                if not is_valid_action(act):
                    raise ValueError(
                        f"Invalid action {act} for key {key!r}"
                    )
                self._key_map[key.lower()] = act

    def key_to_action(self, key: str) -> int:
        """
        Convert a key name to an action index.

        Parameters
        ----------
        key : str
            The key name (case-insensitive).  Common names:
            ``'arrowleft'``, ``'arrowright'``, ``'arrowup'``,
            ``'arrowdown'``, ``' '`` (space).

        Returns
        -------
        int
            The action index (0–4).
        """
        action = self._key_map.get(key.lower(), self._default_action)
        return action

    @property
    def default_action(self) -> int:
        """The action used when no mapping matches."""
        return self._default_action

    @property
    def key_map(self) -> dict[str, int]:
        """The current key→action mapping (copy)."""
        return dict(self._key_map)

    def describe(self) -> str:
        """Return a human-readable description of all key mappings."""
        lines = ["Key Mappings:", "-" * 30]
        for key, act in sorted(self._key_map.items()):
            key_label = key if key.strip() else "Space"
            lines.append(f"  {key_label:15s} → {action_name(act)}")
        lines.append(f"  {'(default)':15s} → {action_name(self._default_action)}")
        return "\n".join(lines)
