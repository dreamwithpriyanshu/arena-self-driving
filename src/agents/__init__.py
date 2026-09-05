"""
src.agents — Reinforcement Learning Agents.

Provides the R (DQN) and S (SARSA) agents. Both implement the BaseAgent
interface. They are completely independent and do not import each other.
"""

from src.agents.base import BaseAgent
from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent

__all__ = ["BaseAgent", "DQNAgent", "SARSAAgent"]
