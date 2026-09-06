"""
Base agent interface.

Defines the interface used by the tabular SARSA policy.

Layer: agents (depends on envs.actions and data.schemas)
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from src.data.schemas import Transition


class BaseAgent(abc.ABC):
    """
    Abstract base class for all reinforcement learning agents.
    """

    @abc.abstractmethod
    def act(self, state: Any, discrete_state: int) -> int:
        """
        Choose an action based on the current state.

        Parameters
        ----------
        state : Any
            The continuous state representation retained in recorded data.
        discrete_state : int
            The discrete state index (used by SARSA).

        Returns
        -------
        int
            The chosen action index.
        """
        pass

    @abc.abstractmethod
    def update(self, transition: Transition) -> dict[str, float]:
        """
        Learn from a single transition.

        Parameters
        ----------
        transition : Transition
            The experience tuple to learn from (or store in replay buffer).

        Returns
        -------
        dict[str, float]
            A dictionary of training metrics (e.g., 'loss', 'q_value') for logging.
            May be empty if no update occurred on this step.
        """
        pass

    @abc.abstractmethod
    def save(self, directory: str | Path) -> None:
        """
        Save the agent's learned weights/tables to a directory.

        Parameters
        ----------
        directory : str or Path
            The directory where artifacts should be saved.
        """
        pass

    @abc.abstractmethod
    def load(self, directory: str | Path) -> None:
        """
        Load the agent's learned weights/tables from a directory.

        Parameters
        ----------
        directory : str or Path
            The directory to load artifacts from.
        """
        pass

    @abc.abstractmethod
    def set_eval_mode(self, eval_mode: bool) -> None:
        """
        Switch between training (exploration) and evaluation (exploitation).

        Parameters
        ----------
        eval_mode : bool
            True for evaluation (no exploration), False for training.
        """
        pass
