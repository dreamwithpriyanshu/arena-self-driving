"""
Training Orchestrator.

Ties together the Environment Manager, the tabular SARSA agent, and the Data Loader.
Responsible for:
1. Loading arrow-key demonstrations and warm-starting SARSA.
2. Running autonomous training loops.
3. Running evaluation loops.
4. Managing unified checkpoints.

Layer: training (depends on simulation, agents, data; knows nothing about UI)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from src.agents.sarsa import SARSAAgent
from src.data.loader import load_all_episodes
from src.data.schemas import Transition
from src.simulation.env_manager import EnvManager

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the training and evaluation of autonomous agents.
    """

    def __init__(
        self,
        sarsa_agent: SARSAAgent,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:
        self.sarsa = sarsa_agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def warm_start(self, data_dir: str | Path = "data/human_demonstrations") -> dict[str, int]:
        """
        Load human demonstrations and warm-start SARSA from their trajectories.

        Returns
        -------
        dict
            Counts of transitions loaded and processed.
        """
        logger.info("Starting warm-start from %s", data_dir)
        
        try:
            episodes = load_all_episodes(data_dir, validate=True, skip_invalid=True)
        except Exception as e:
            logger.error("Failed to load demonstrations: %s", e)
            return {"loaded": 0, "sarsa_warm_started": 0}

        if not episodes:
            logger.warning("No valid transitions found for warm start.")
            return {"loaded": 0, "sarsa_warm_started": 0}

        transitions = [transition for _metadata, episode in episodes for transition in episode]
        for _metadata, episode in episodes:
            for index, transition in enumerate(episode):
                next_action = episode[index + 1].action if index + 1 < len(episode) else None
                self.sarsa.update(transition, next_action=next_action)

        return {
            "loaded": len(transitions),
            "sarsa_warm_started": len(transitions),
        }

    def train_episode(
        self,
        env_mgr: EnvManager,
        seed: Optional[int] = None,
        max_steps: int = 1000,
        step_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Run one autonomous SARSA training episode.

        Parameters
        ----------
        env_mgr : EnvManager
            The environment manager instance to step through.
        seed : int, optional
            Environment seed.
        max_steps : int
            Maximum steps before truncating.
        step_callback: Callable, optional
            A function called at each step with (result, metrics) for live UI rendering.

        Returns
        -------
        dict
            Episode metrics (reward, steps, losses, etc.)
        """
        self.sarsa.set_eval_mode(False)

        result = env_mgr.reset(seed=seed)
        
        episode_td_error = 0.0
        episode_q_val = 0.0
        update_count = 0

        start_time = time.time()

        action = self.sarsa.act(state=result.raw_state, discrete_state=result.discrete_state)
        for step in range(max_steps):

            prev_result = result
            result = env_mgr.step(action)
            
            if step >= max_steps - 1 and not result.terminated:
                result.truncated = True

            transition = Transition(
                step=step,
                state=prev_result.raw_state.tolist(),
                action=action,
                reward=result.reward,
                next_state=result.raw_state.tolist(),
                terminated=result.terminated,
                truncated=result.truncated,
                discrete_state=prev_result.discrete_state,
                next_discrete_state=result.discrete_state,
                lane=result.lane_index,
                speed=result.speed,
            )

            next_action = None
            if not (result.terminated or result.truncated):
                next_action = self.sarsa.act(state=result.raw_state, discrete_state=result.discrete_state)
            metrics = self.sarsa.update(transition, next_action=next_action)
            
            if metrics:
                update_count += 1
                if "td_error" in metrics:
                    episode_td_error += metrics["td_error"]
                if "avg_q" in metrics:
                    episode_q_val += metrics["avg_q"]

            # Callback for optional CLI progress reporting.
            if step_callback:
                step_callback(result, metrics)

            if result.terminated or result.truncated:
                break
            action = next_action

        duration = time.time() - start_time

        summary = {
            "vehicle": "SARSA",
            "steps": env_mgr.step_count,
            "total_reward": env_mgr.total_reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "duration": duration,
        }
        
        if update_count > 0:
            summary["avg_td_error"] = episode_td_error / update_count
            summary["epsilon"] = self.sarsa.epsilon
            summary["avg_q"] = episode_q_val / update_count

        return summary



    def evaluate_episode(
        self,
        env_mgr: EnvManager,
        seed: Optional[int] = None,
        max_steps: int = 1000,
        step_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Run one autonomous evaluation episode (greedy policy, no learning).
        """
        self.sarsa.set_eval_mode(True)

        result = env_mgr.reset(seed=seed)
        start_time = time.time()

        for step in range(max_steps):
            action = self.sarsa.act(state=result.raw_state, discrete_state=result.discrete_state)
            result = env_mgr.step(action)
            
            if step >= max_steps - 1 and not result.terminated:
                result.truncated = True

            if step_callback:
                step_callback(result, {})

            if result.terminated or result.truncated:
                break

        duration = time.time() - start_time

        return {
            "vehicle": "SARSA",
            "steps": env_mgr.step_count,
            "total_reward": env_mgr.total_reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "duration": duration,
        }

    def save_checkpoints(self) -> None:
        """Save the SARSA Q-table to the configured checkpoint directory."""
        logger.info("Saving checkpoints to %s", self.checkpoint_dir)
        self.sarsa.save(self.checkpoint_dir)

    def load_checkpoints(self) -> None:
        """Load the SARSA Q-table from the configured checkpoint directory."""
        logger.info("Loading checkpoints from %s", self.checkpoint_dir)
        self.sarsa.load(self.checkpoint_dir)
