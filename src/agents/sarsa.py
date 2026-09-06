"""
SARSA Agent (State-Action-Reward-State-Action).

Implementation of tabular SARSA using numpy. Uses a discretised state
index and maintains a Q-table. Features an epsilon-greedy policy and
on-policy updates.

Layer: agents (depends on envs, data.schemas; isolated from DQN)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.base import BaseAgent
from src.data.schemas import Transition
from src.envs.actions import NUM_ACTIONS
from src.envs.state_builder import total_discrete_states

logger = logging.getLogger(__name__)


class SARSAAgent(BaseAgent):
    """
    Tabular SARSA Agent.
    """

    def __init__(
        self,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.num_states = total_discrete_states()
        self.num_actions = NUM_ACTIONS
        self.lr = lr
        self.gamma = gamma

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._eval_mode = False

        # Q-table initialized to zeros
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def act(self, state: Any, discrete_state: int) -> int:
        """Epsilon-greedy action selection."""
        # Note: discrete_state must be within [0, self.num_states)
        if not (0 <= discrete_state < self.num_states):
            logger.warning(
                "SARSA discrete_state %d out of bounds. Defaulting to IDLE.",
                discrete_state
            )
            return 1  # IDLE

        if self._eval_mode or random.random() > self.epsilon:
            # Exploit: choose action with max Q-value. Break ties randomly.
            q_values = self.q_table[discrete_state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(random.choice(best_actions))
        else:
            # Explore
            return random.randrange(self.num_actions)

    def update(
        self,
        transition: Transition,
        next_action: int | None = None,
    ) -> dict[str, float]:
        """
        Perform a single SARSA update:
        Q(S, A) <- Q(S, A) + lr * [R + gamma * Q(S', A') - Q(S, A)]
        """
        metrics: dict[str, float] = {}
        if self._eval_mode:
            return metrics

        s = transition.discrete_state
        a = transition.action
        r = transition.reward
        s_next = transition.next_discrete_state
        
        if transition.terminated or transition.truncated:
            q_next = 0.0
        else:
            # Online training supplies the exact action that will be executed
            # at S'.  This is the defining on-policy SARSA update.  For older
            # demonstration files, use the current policy as a safe fallback.
            a_next = self.act(None, s_next) if next_action is None else next_action
            if not 0 <= a_next < self.num_actions:
                raise ValueError(f"next_action must be in 0..{self.num_actions - 1}")
            q_next = self.q_table[s_next, a_next]

        # Bound checks
        if 0 <= s < self.num_states and 0 <= a < self.num_actions:
            current_q = self.q_table[s, a]
            target = r + self.gamma * q_next
            td_error = target - current_q
            
            # Update table
            self.q_table[s, a] += self.lr * td_error
            
            metrics["td_error"] = float(td_error)
            metrics["avg_q"] = float(np.mean(self.q_table))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        metrics["epsilon"] = self.epsilon

        return metrics

    def warm_start(self, transitions: list[Transition]) -> None:
        """
        Pre-train the Q-table using human demonstrations.
        Iterates over transitions and performs tabular updates.
        """
        count = 0
        for t in transitions:
            self.update(t)
            count += 1
        logger.info("SARSAAgent: Warm-started on %d transitions.", count)

    def set_eval_mode(self, eval_mode: bool) -> None:
        self._eval_mode = eval_mode
        logger.info("SARSAAgent: eval_mode=%s", eval_mode)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        table_path = directory / "sarsa_q_table.npy"
        np.save(table_path, self.q_table)
        logger.info("SARSAAgent saved to %s", table_path)

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        table_path = directory / "sarsa_q_table.npy"
        if not table_path.exists():
            raise FileNotFoundError(f"SARSA table not found: {table_path}")
        
        # allow_pickle=False is the default in modern numpy, but explicit is safer
        self.q_table = np.load(table_path, allow_pickle=False)
        
        # Verify shape
        expected_shape = (self.num_states, self.num_actions)
        if self.q_table.shape != expected_shape:
            raise ValueError(
                f"Loaded Q-table shape {self.q_table.shape} does not match "
                f"expected {expected_shape}."
            )
        logger.info("SARSAAgent loaded from %s", table_path)
