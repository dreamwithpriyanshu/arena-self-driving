"""
Training Facade.

Provides a strict UI-facing wrapper around the `TrainingOrchestrator`.
Per architecture rules (§4), Streamlit pages MUST NOT import `src.training`,
`src.agents`, or `src.envs` directly. This facade instantiates the agents and
orchestrator, and exposes only the high-level commands the UI needs.

Layer: simulation (depends on training, agents; UI depends on this)
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

# Safe to import here (behind the facade)
from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.simulation.env_manager import EnvManager
from src.training.orchestrator import TrainingOrchestrator


class TrainingFacade:
    """
    Singleton-style manager to hold the training orchestrator state 
    for the Streamlit application.
    """

    def __init__(self, checkpoint_dir: str | Path = "artifacts/checkpoints") -> None:
        self.dqn = DQNAgent()
        self.sarsa = SARSAAgent()
        self.orchestrator = TrainingOrchestrator(
            dqn_agent=self.dqn,
            sarsa_agent=self.sarsa,
            checkpoint_dir=checkpoint_dir,
        )

    def warm_start(self, data_dir: str | Path = "data/human_demonstrations") -> dict[str, int]:
        """Warm-start both agents from human demonstrations."""
        return self.orchestrator.warm_start(data_dir=data_dir)

    def train_episode(
        self,
        vehicle: str,
        env_mgr: EnvManager,
        seed: Optional[int] = None,
        max_steps: int = 1000,
        step_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Run a single training episode."""
        return self.orchestrator.train_episode(
            vehicle=vehicle, env_mgr=env_mgr, seed=seed, max_steps=max_steps, step_callback=step_callback
        )

    def evaluate_episode(
        self,
        vehicle: str,
        env_mgr: EnvManager,
        seed: Optional[int] = None,
        max_steps: int = 1000,
        step_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Run a single evaluation episode (greedy)."""
        return self.orchestrator.evaluate_episode(
            vehicle=vehicle, env_mgr=env_mgr, seed=seed, max_steps=max_steps, step_callback=step_callback
        )

    def save_checkpoints(self) -> None:
        """Save the models."""
        self.orchestrator.save_checkpoints()

    def load_checkpoints(self) -> None:
        """Load the models."""
        self.orchestrator.load_checkpoints()

    # --- Read-only properties for UI display ---

    @property
    def dqn_epsilon(self) -> float:
        return self.dqn.epsilon

    @property
    def sarsa_epsilon(self) -> float:
        return self.sarsa.epsilon
