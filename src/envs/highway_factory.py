"""
HighwayEnv factory — creates and configures simulation environments.

This module is the *only* place that knows about ``gymnasium.make`` and
HighwayEnv config dictionaries.  Everything else receives a ready-to-use
``gymnasium.Env`` instance through the simulation façade.

Layer: envs  (knows nothing about agents, training, or Streamlit)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import yaml

# Ensure HighwayEnv environments are registered with Gymnasium.
import highway_env  # noqa: F401 (Registers the environments with gymnasium)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default_env.yaml"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and return a YAML config file as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_configs(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge *overrides* into *base*, returning a new dict.

    Nested dicts are merged; other types are replaced.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_default_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Load the default environment configuration from YAML.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config file.  Falls back to
        ``configs/default_env.yaml`` in the project root, or the path in
        the ``ENV_CONFIG_PATH`` environment variable.

    Returns
    -------
    dict
        The parsed configuration dictionary.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Environment config not found at {config_path}. "
            "Make sure configs/default_env.yaml exists."
        )
    return _load_yaml_config(config_path)


def _build_gym_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Convert our YAML-friendly config into the dict HighwayEnv expects.

    HighwayEnv reads a flat config dict where observation/action/reward
    keys are nested dicts.  Our YAML is already structured that way, so
    this is mostly pass-through with a few reshuffles.
    """
    gym_cfg: dict[str, Any] = {}

    # Road / traffic
    gym_cfg["lanes_count"] = cfg.get("lanes_count", 4)
    gym_cfg["vehicles_count"] = cfg.get("vehicles_count", 15)
    gym_cfg["vehicles_density"] = cfg.get("vehicles_density", 1.0)
    gym_cfg["controlled_vehicles"] = cfg.get("controlled_vehicles", 1)
    if cfg.get("initial_lane_id") is not None:
        gym_cfg["initial_lane_id"] = cfg["initial_lane_id"]

    # Observation
    obs = cfg.get("observation", {})
    gym_cfg["observation"] = {
        "type": obs.get("type", "Kinematics"),
        "vehicles_count": obs.get("vehicles_count", 6),
        "features": obs.get("features", ["x", "y", "vx", "vy", "cos_h", "sin_h"]),
        "absolute": obs.get("absolute", False),
        "normalize": obs.get("normalize", True),
    }
    if "observation_config" in obs:
        gym_cfg["observation"]["observation_config"] = obs["observation_config"]

    # Action
    act = cfg.get("action", {})
    gym_cfg["action"] = {
        "type": act.get("type", "DiscreteMetaAction"),
    }
    if "action_config" in act:
        gym_cfg["action"]["action_config"] = act["action_config"]

    # Reward
    rew = cfg.get("reward", {})
    gym_cfg["collision_reward"] = rew.get("collision_reward", -1.0)
    gym_cfg["right_lane_reward"] = rew.get("right_lane_reward", 0.1)
    gym_cfg["high_speed_reward"] = rew.get("high_speed_reward", 0.4)
    gym_cfg["lane_change_reward"] = rew.get("lane_change_reward", 0.0)
    if "reward_speed_range" in rew:
        gym_cfg["reward_speed_range"] = rew["reward_speed_range"]

    # Timing
    gym_cfg["simulation_frequency"] = cfg.get("simulation_frequency", 15)
    gym_cfg["policy_frequency"] = cfg.get("policy_frequency", 5)
    gym_cfg["duration"] = cfg.get("duration", 40)

    # Rendering
    gym_cfg["screen_width"] = cfg.get("screen_width", 600)
    gym_cfg["screen_height"] = cfg.get("screen_height", 300)

    return gym_cfg


def create_highway_env(
    config_overrides: Optional[dict[str, Any]] = None,
    render_mode: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> gym.Env:
    """
    Create and return a configured HighwayEnv instance.

    Parameters
    ----------
    config_overrides : dict, optional
        Key-value pairs that override the YAML defaults (merged recursively).
    render_mode : str, optional
        Gymnasium render mode.  If ``None``, uses the value from the config
        file (default ``"rgb_array"``).
    config_path : str or Path, optional
        Alternate YAML config path (defaults to ``configs/default_env.yaml``).

    Returns
    -------
    gymnasium.Env
        A fully configured ``highway-v0`` environment.

    Examples
    --------
    >>> env = create_highway_env()
    >>> obs, info = env.reset()
    >>> obs.shape
    (6, 6)
    """
    base_cfg = load_default_config(config_path)
    if config_overrides:
        base_cfg = _merge_configs(base_cfg, config_overrides)

    if render_mode is None:
        render_mode = base_cfg.get("render_mode", "rgb_array")

    gym_cfg = _build_gym_config(base_cfg)

    env = gym.make("highway-v0", render_mode=render_mode, config=gym_cfg)
    return env
