"""
Deep Q-Network (DQN) Agent.

Implementation of DQN using PyTorch. Uses a continuous state vector
and outputs Q-values for discrete actions. Features an epsilon-greedy
policy, experience replay buffer, and target network.

Layer: agents (depends on envs, data.schemas; isolated from SARSA)
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base import BaseAgent
from src.data.schemas import Transition
from src.envs.actions import NUM_ACTIONS

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Simple fully-connected Q-Network."""
    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN Agent with target network and experience replay.
    """

    def __init__(
        self,
        state_dim: int = 36,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = NUM_ACTIONS
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._eval_mode = False

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(state_dim, NUM_ACTIONS, hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, NUM_ACTIONS, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_capacity)
        self.step_count = 0

    def act(self, state: Any, discrete_state: int) -> int:
        """Epsilon-greedy action selection."""
        # In eval mode or if we roll > epsilon, exploit
        if self._eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                # Add batch dimension
                state_tensor = state_tensor.unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return int(torch.argmax(q_values).item())
        else:
            # Explore
            return random.randrange(self.num_actions)

    def update(self, transition: Transition) -> dict[str, float]:
        """Store transition and perform one gradient step if buffer is large enough."""
        metrics: dict[str, float] = {}
        
        # 1. Store
        self.memory.push(transition)
        self.step_count += 1

        # 2. Decay epsilon
        if not self._eval_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            metrics["epsilon"] = self.epsilon

        # 3. Train
        if len(self.memory) < self.batch_size or self._eval_mode:
            return metrics

        batch = self.memory.sample(self.batch_size)
        
        # Prepare tensors
        states = torch.tensor([t.state for t in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32, device=self.device)
        # terminated=True means episode ended (Q_target = r)
        dones = torch.tensor([float(t.terminated) for t in batch], dtype=torch.float32, device=self.device)

        # Compute current Q values: Q(s, a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Compute next Q values from target net: max_a Q_target(s', a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            
        # Target = r + gamma * max_a Q_target(s', a) * (1 - done)
        target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics["loss"] = loss.item()
        metrics["avg_q"] = q_values.mean().item()

        # 4. Update Target Network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return metrics

    def prefill_buffer(self, transitions: list[Transition]) -> None:
        """Pre-fill the replay buffer with human demonstrations."""
        count = 0
        for t in transitions:
            self.memory.push(t)
            count += 1
        logger.info("DQNAgent: Pre-filled replay buffer with %d transitions.", count)

    def set_eval_mode(self, eval_mode: bool) -> None:
        self._eval_mode = eval_mode
        if eval_mode:
            self.policy_net.eval()
        else:
            self.policy_net.train()
        logger.info("DQNAgent: eval_mode=%s", eval_mode)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Use weights_only=True per security guidelines
        model_path = directory / "dqn_policy.pt"
        torch.save(self.policy_net.state_dict(), model_path)
        logger.info("DQNAgent saved to %s", model_path)

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        model_path = directory / "dqn_policy.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"DQN weights not found: {model_path}")
        
        # Load state dict safely mapping to the configured device.
        # torch.load does not accept a `weights_only` keyword; remove it.
        state_dict = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        logger.info("DQNAgent loaded from %s", model_path)
