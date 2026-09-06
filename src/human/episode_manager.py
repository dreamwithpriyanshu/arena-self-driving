"""
Episode manager — orchestrates a human-driven episode from start to finish.

Ties together the environment, keyboard controller, and transition recorder
for the native PyGame human-driving runner.

Layer: human  (depends on simulation, data, envs.actions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.data.recorder import TransitionRecorder
from src.data.schemas import Transition
from src.human.keyboard_controller import KeyboardController
from src.envs.actions import validate_action
from src.simulation.env_manager import EnvManager, StepResult

logger = logging.getLogger(__name__)


class EpisodeManager:
    """
    Manages the lifecycle of a single human-driven episode.

    Workflow::

        mgr = EpisodeManager()
        mgr.start(vehicle="S", seed=42)

        # In a loop driven by the UI:
        result = mgr.act("arrowup")       # human presses Up
        result = mgr.act("arrowleft")     # human presses Left
        ...

        if mgr.is_done:
            saved_path = mgr.save()       # or mgr.discard()

    The manager ensures that every transition is recorded and that
    the environment is properly cleaned up.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/human_demonstrations",
        config_overrides: Optional[dict[str, Any]] = None,
        render_mode: str = "rgb_array",
        max_steps: int = 1000,
    ) -> None:
        self._base_dir = base_dir
        self._config_overrides = config_overrides or {}
        self._render_mode = render_mode
        self._max_steps = max_steps

        self._env_mgr: Optional[EnvManager] = None
        self._recorder: Optional[TransitionRecorder] = None
        self._keyboard = KeyboardController()

        self._prev_result: Optional[StepResult] = None
        self._episode_active: bool = False
        self._episode_done: bool = False
        self._vehicle: str = "S"

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        vehicle: str = "S",
        seed: Optional[int] = None,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        """
        Start a new human-driven episode.

        Parameters
        ----------
        vehicle : str
            ``'S'`` — the SARSA driving label.
        seed : int, optional
            Environment random seed.
        config_overrides : dict, optional
            Additional env config overrides for this episode.

        Returns
        -------
        StepResult
            The initial observation after reset.
        """
        if self._episode_active:
            raise RuntimeError(
                "An episode is already active. "
                "Call save() or discard() first."
            )

        if vehicle != "S":
            raise ValueError(f"vehicle must be 'S', got {vehicle!r}")

        self._vehicle = vehicle

        # Merge config overrides
        merged_config = dict(self._config_overrides)
        if config_overrides:
            merged_config.update(config_overrides)

        # Create environment manager
        self._env_mgr = EnvManager(
            config_overrides=merged_config,
            render_mode=self._render_mode,
        )

        # Create recorder
        self._recorder = TransitionRecorder(
            base_dir=self._base_dir,
            source="human",
        )

        # Reset environment
        result = self._env_mgr.reset(seed=seed)
        initial_speed = merged_config.get("initial_speed")
        if initial_speed is not None:
            for controlled_vehicle in getattr(self._env_mgr._env.unwrapped, "controlled_vehicles", []):
                controlled_vehicle.speed = float(initial_speed)
                controlled_vehicle.target_speed = float(initial_speed)
        self._prev_result = result
        self._episode_active = True
        self._episode_done = False

        # Start recording
        self._recorder.start_episode(
            vehicle=vehicle,
            seed=seed,
            config_overrides=merged_config,
        )

        logger.info(
            "Human episode started: vehicle=%s, seed=%s, max_steps=%d",
            vehicle, seed, self._max_steps,
        )
        return result

    def act(self, key: str) -> StepResult:
        """
        Process a single human key press: translate to action, step env,
        record the transition.

        Parameters
        ----------
        key : str
            The key name from the keyboard event.

        Returns
        -------
        StepResult
            The result after taking the action.

        Raises
        ------
        RuntimeError
            If no episode is active or the episode is already done.
        """
        return self.act_action(self._keyboard.key_to_action(key))

    def set_max_steps(self, max_steps: int) -> None:
        """Update the episode limit before it reaches its current boundary."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._max_steps = max_steps

    def act_action(self, action: int) -> StepResult:
        """Apply a validated discrete action and record its transition."""
        if not self._episode_active:
            raise RuntimeError("No episode active. Call start() first.")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call save() or discard().")
        if self._env_mgr is None or self._recorder is None:
            raise RuntimeError("Internal error: env_mgr or recorder is None.")
        if self._prev_result is None:
            raise RuntimeError("Internal error: no previous result.")

        action = validate_action(action)

        # Step the environment
        result = self._env_mgr.step(action)
        
        # Enforce max steps if environment didn't already truncate
        if self._env_mgr.step_count >= self._max_steps and not result.terminated:
            result.truncated = True

        # Record the transition
        transition = Transition(
            step=self._env_mgr.step_count - 1,  # 0-based
            state=self._prev_result.raw_state.tolist(),
            action=action,
            reward=result.reward,
            next_state=result.raw_state.tolist(),
            terminated=result.terminated,
            truncated=result.truncated,
            discrete_state=self._prev_result.discrete_state,
            next_discrete_state=result.discrete_state,
            lane=result.lane_index,
            speed=result.speed,
        )
        self._recorder.record(transition)

        # Update state
        self._prev_result = result
        if result.terminated or result.truncated:
            self._episode_done = True
            logger.info(
                "Episode ended after %d steps (reward=%.2f, truncated=%s)",
                self._env_mgr.step_count,
                self._env_mgr.total_reward,
                result.truncated,
            )

        return result

    def save(self) -> Path:
        """
        Save the current episode and clean up.

        Returns
        -------
        Path
            Path to the saved JSONL file.
        """
        if not self._episode_active:
            raise RuntimeError("No episode active.")
        if self._recorder is None:
            raise RuntimeError("Internal error: recorder is None.")

        filepath = self._recorder.save_episode()
        self._cleanup()

        logger.info("Episode saved: %s", filepath)
        return filepath

    def discard(self) -> None:
        """Discard the current episode and clean up."""
        if not self._episode_active:
            raise RuntimeError("No episode active.")
        if self._recorder is not None:
            self._recorder.discard_episode()

        self._cleanup()
        logger.info("Episode discarded.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if an episode is in progress."""
        return self._episode_active

    @property
    def is_done(self) -> bool:
        """True if the episode ended (collision/timeout) but hasn't been saved/discarded."""
        return self._episode_done

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        if self._env_mgr is None:
            return 0
        return self._env_mgr.step_count

    @property
    def total_reward(self) -> float:
        """Cumulative reward in the current episode."""
        if self._env_mgr is None:
            return 0.0
        return self._env_mgr.total_reward

    @property
    def vehicle(self) -> str:
        """The SARSA model label (always ``'S'``)."""
        return self._vehicle

    @property
    def keyboard(self) -> KeyboardController:
        """The keyboard controller (for displaying mappings in UI)."""
        return self._keyboard



    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Release resources."""
        if self._env_mgr is not None:
            self._env_mgr.close()
            self._env_mgr = None
        self._recorder = None
        self._prev_result = None
        self._episode_active = False
        self._episode_done = False
