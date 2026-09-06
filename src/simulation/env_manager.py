"""
Environment Manager — the simulation façade.

This is the **only** module that Streamlit pages are allowed to import
from the ``src`` package.  It wraps the ``envs`` layer and exposes a
simple interface for resetting, stepping, reading state, and closing
the environment.

Layer: simulation (shared by native playback, human recording, and training)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.envs.actions import NUM_ACTIONS, action_name, validate_action
from src.envs.highway_factory import create_highway_env
from src.envs.state_builder import (
    build_discrete_state_with_lane,
    build_raw_state,
    total_discrete_states,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Immutable snapshot returned after every ``env.step()``."""

    raw_obs: np.ndarray
    """Raw Kinematics matrix (V, F) straight from the env."""

    raw_state: np.ndarray
    """Flattened continuous state vector retained for recorded transitions."""

    discrete_state: int
    """Integer state index for tabular SARSA."""

    reward: float
    """Scalar reward from the environment."""

    terminated: bool
    """True if the episode ended (collision or timeout)."""

    truncated: bool
    """True if the episode was truncated by Gymnasium."""

    info: dict[str, Any] = field(default_factory=dict)
    """Extra info dict from Gymnasium."""

    action_taken: Optional[int] = None
    """The action that produced this result (populated by step())."""

    lane_index: int = 0
    """Ego vehicle's current lane index."""

    speed: float = 0.0
    """Ego vehicle's current speed (m/s)."""



# Manager
# ---------------------------------------------------------------------------

class EnvManager:
    """
    Thin wrapper around the HighwayEnv environment.

    Usage::

        mgr = EnvManager()
        result = mgr.reset()
        while not result.terminated:
            result = mgr.step(action=3)  # FASTER
        mgr.close()
    """

    def __init__(
        self,
        config_overrides: Optional[dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self._config_overrides = config_overrides or {}
        self._render_mode = render_mode
        self._env = None
        self._last_obs: Optional[np.ndarray] = None
        self._last_info: dict[str, Any] = {}
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._lanes_count: int = self._config_overrides.get("lanes_count", 4)
        self._is_open: bool = False

        logger.info("EnvManager created (not yet reset).")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> StepResult:
        """
        Create (or recreate) the environment and reset it.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible episodes.

        Returns
        -------
        StepResult
            The initial observation/state after reset.
        """
        self.close()  # clean up any previous env

        self._env = create_highway_env(
            config_overrides=self._config_overrides,
            render_mode=self._render_mode,
        )
        self._is_open = True

        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs, info = self._env.reset(**reset_kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = info
        self._step_count = 0
        self._total_reward = 0.0

        lane_index = self._extract_lane_index(info)
        speed = self._extract_speed(info)

        result = StepResult(
            raw_obs=self._last_obs,
            raw_state=build_raw_state(self._last_obs),
            discrete_state=build_discrete_state_with_lane(
                self._last_obs, lane_index, lanes_count=self._lanes_count,
            ),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            action_taken=None,
            lane_index=lane_index,
            speed=speed,
        )
        logger.info("Environment reset (seed=%s).", seed)
        return result

    def step(self, action: int) -> StepResult:
        """
        Advance the environment by one decision step.

        Parameters
        ----------
        action : int
            A valid action index (0–4).

        Returns
        -------
        StepResult
            The resulting observation, state, reward, and done flags.

        Raises
        ------
        RuntimeError
            If the environment has not been reset.
        ValueError
            If *action* is not a valid action index.
        """
        if self._env is None or not self._is_open:
            raise RuntimeError(
                "Environment is not open. Call reset() before step()."
            )
        validate_action(action)

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = info
        self._step_count += 1
        self._total_reward += reward

        lane_index = self._extract_lane_index(info)
        speed = self._extract_speed(info)

        result = StepResult(
            raw_obs=self._last_obs,
            raw_state=build_raw_state(self._last_obs),
            discrete_state=build_discrete_state_with_lane(
                self._last_obs, lane_index, lanes_count=self._lanes_count,
            ),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
            action_taken=action,
            lane_index=lane_index,
            speed=speed,
        )

        if terminated or truncated:
            logger.info(
                "Episode ended after %d steps (total_reward=%.2f, "
                "terminated=%s, truncated=%s).",
                self._step_count, self._total_reward, terminated, truncated,
            )

        return result

    def close(self) -> None:
        """Release environment resources safely."""
        if self._env is not None and self._is_open:
            try:
                self._env.close()
                logger.info("Environment closed.")
            except Exception:
                logger.exception("Error closing environment.")
            finally:
                self._is_open = False
                self._env = None

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    @property
    def total_reward(self) -> float:
        """Cumulative reward in the current episode."""
        return self._total_reward

    @property
    def is_open(self) -> bool:
        """True if the environment is currently active."""
        return self._is_open

    @property
    def num_actions(self) -> int:
        """Number of discrete actions available."""
        return NUM_ACTIONS

    @property
    def num_discrete_states(self) -> int:
        """Total number of discrete states for SARSA."""
        return total_discrete_states(lanes_count=self._lanes_count)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_lane_index(self, info: dict[str, Any]) -> int:
        """
        Best-effort extraction of the ego vehicle's lane index.

        HighwayEnv stores the controlled vehicle at
        ``env.unwrapped.vehicle``.
        """
        try:
            if self._env is not None:
                vehicle = getattr(self._env.unwrapped, "vehicle", None)
                if vehicle is not None and hasattr(vehicle, "lane_index"):
                    li = vehicle.lane_index
                    if isinstance(li, (tuple, list)):
                        return int(li[-1])
                    return int(li)
        except (TypeError, ValueError, IndexError, AttributeError):
            pass
        return 0

    def _extract_speed(self, info: dict[str, Any]) -> float:
        """
        Best-effort extraction of the ego vehicle's speed.

        HighwayEnv stores the controlled vehicle at
        ``env.unwrapped.vehicle``.
        """
        try:
            if self._env is not None:
                vehicle = getattr(self._env.unwrapped, "vehicle", None)
                if vehicle is not None and hasattr(vehicle, "speed"):
                    return float(vehicle.speed)
        except (TypeError, ValueError, AttributeError):
            pass
        return 0.0
