"""
src.envs — Environment layer.

Provides the HighwayEnv factory, shared action set, and state builders.
This layer knows nothing about agents, training, or Streamlit.
"""

from src.envs.actions import (
    Action,
    ACTION_INDEX,
    ACTION_NAMES,
    KEYBOARD_ACTION_MAP,
    NUM_ACTIONS,
    action_name,
    is_valid_action,
    keyboard_to_action,
    validate_action,
)
from src.envs.highway_factory import create_highway_env, load_default_config
from src.envs.state_builder import (
    build_discrete_state,
    build_discrete_state_with_lane,
    build_raw_state,
    build_raw_state_from_env,
    raw_state_dim,
    total_discrete_states,
)

__all__ = [
    "Action",
    "ACTION_INDEX",
    "ACTION_NAMES",
    "KEYBOARD_ACTION_MAP",
    "NUM_ACTIONS",
    "action_name",
    "is_valid_action",
    "keyboard_to_action",
    "validate_action",
    "create_highway_env",
    "load_default_config",
    "build_discrete_state",
    "build_discrete_state_with_lane",
    "build_raw_state",
    "build_raw_state_from_env",
    "raw_state_dim",
    "total_discrete_states",
]
