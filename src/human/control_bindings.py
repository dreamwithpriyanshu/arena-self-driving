"""Load and resolve native PyGame control profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.envs.actions import Action


CONTROL_BINDINGS_PATH = Path(__file__).resolve().parents[2] / "configs" / "control_bindings.json"
_VALID_KEYS = {"left", "right", "up", "down", "space"}


def load_control_profiles(path: str | Path = CONTROL_BINDINGS_PATH) -> tuple[str, dict[str, dict[Action, tuple[str, ...]]]]:
    """Return the default profile and validated action-to-key mappings."""
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    profiles_raw = raw.get("profiles")
    default_profile = raw.get("default_profile")
    if not isinstance(profiles_raw, dict) or not isinstance(default_profile, str):
        raise ValueError("control_bindings.json must define profiles and default_profile")
    if default_profile not in profiles_raw:
        raise ValueError("default_profile must name a configured profile")

    profiles: dict[str, dict[Action, tuple[str, ...]]] = {}
    for profile_name, mapping in profiles_raw.items():
        if not isinstance(profile_name, str) or not isinstance(mapping, dict):
            raise ValueError("Each control profile must be an action mapping")
        parsed: dict[Action, tuple[str, ...]] = {}
        for action in Action:
            keys = mapping.get(action.name, [])
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise ValueError(f"{profile_name}.{action.name} must be a list of key names")
            unknown = set(keys) - _VALID_KEYS
            if unknown:
                raise ValueError(f"{profile_name}.{action.name} has unsupported keys: {sorted(unknown)}")
            parsed[action] = tuple(keys)
        profiles[profile_name] = parsed
    return default_profile, profiles


def active_actions(profile: dict[Action, tuple[str, ...]], pressed_keys: Iterable[str]) -> list[Action]:
    """Return every non-idle action currently held by the driver."""
    pressed = set(pressed_keys)
    return [
        action
        for action in (Action.LANE_LEFT, Action.LANE_RIGHT, Action.FASTER, Action.SLOWER)
        if pressed.intersection(profile[action])
    ]
