"""
Human Training Facade.

Provides a strict UI-facing wrapper around the `EpisodeManager`.
Per architecture rules (§4), Streamlit pages MUST NOT import `src.human` directly.

Layer: simulation (depends on human; UI depends on this)
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

# Safe to import here (behind the facade)
from src.human.episode_manager import EpisodeManager


class HumanFacade:
    """
    Wrapper for the human EpisodeManager to be used by the UI.
    """

    def __init__(self, base_dir: str | Path = "data/human_demonstrations") -> None:
        self.manager = EpisodeManager(base_dir=base_dir, render_mode="rgb_array")

    def start(self, vehicle: str = "R", seed: Optional[int] = None) -> Any:
        return self.manager.start(vehicle=vehicle, seed=seed)

    def act(self, key: str) -> Any:
        return self.manager.act(key)

    def save(self) -> str:
        return self.manager.save()

    def discard(self) -> None:
        self.manager.discard()

    @property
    def episode_active(self) -> bool:
        return self.manager._episode_active

    @property
    def episode_done(self) -> bool:
        return self.manager._episode_done

    @property
    def step_count(self) -> int:
        if self.manager._env_mgr:
            return self.manager._env_mgr.step_count
        return 0

    @property
    def total_reward(self) -> float:
        if self.manager._env_mgr:
            return self.manager._env_mgr.total_reward
        return 0.0

    @property
    def raw_state(self) -> Optional[Any]:
        if self.manager._prev_result:
            return self.manager._prev_result.raw_state
        return None
