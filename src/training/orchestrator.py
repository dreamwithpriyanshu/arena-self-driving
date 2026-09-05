"""
Training Orchestrator.

Ties together the Environment Manager, the RL Agents (DQN & SARSA), and the Data Loader.
Responsible for:
1. Loading human demonstrations and warm-starting both agents.
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

from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.data.loader import load_all_transitions
from src.data.schemas import Transition
from src.simulation.env_manager import EnvManager

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the training and evaluation of autonomous agents.
    """

    def __init__(
        self,
        dqn_agent: DQNAgent,
        sarsa_agent: SARSAAgent,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:
        self.dqn = dqn_agent
        self.sarsa = sarsa_agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def warm_start(self, data_dir: str | Path = "data/human_demonstrations") -> dict[str, int]:
        """
        Load human demonstrations and feed them to both agents.

        Returns
        -------
        dict
            Counts of transitions loaded and processed.
        """
        logger.info("Starting warm-start from %s", data_dir)
        
        try:
            transitions = load_all_transitions(data_dir, validate=True, skip_invalid=True)
        except Exception as e:
            logger.error("Failed to load demonstrations: %s", e)
            return {"loaded": 0, "r_prefilled": 0, "s_warm_started": 0}

        if not transitions:
            logger.warning("No valid transitions found for warm start.")
            return {"loaded": 0, "r_prefilled": 0, "s_warm_started": 0}

        # DQN uses all transitions for its replay buffer
        self.dqn.prefill_buffer(transitions)
        
        # SARSA iterates through all transitions and performs on-policy updates
        self.sarsa.warm_start(transitions)

        return {
            "loaded": len(transitions),
            "r_prefilled": len(transitions),
            "s_warm_started": len(transitions),
        }

    def train_episode(
        self,
        vehicle: str,
        env_mgr: EnvManager,
        seed: Optional[int] = None,
        max_steps: int = 1000,
        step_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Run one autonomous training episode for the specified vehicle.

        Parameters
        ----------
        vehicle : str
            'R' for DQN, 'S' for SARSA.
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
        if vehicle not in ("R", "S"):
            raise ValueError(f"Vehicle must be 'R' or 'S', got {vehicle!r}")

        agent = self.dqn if vehicle == "R" else self.sarsa
        agent.set_eval_mode(False)

        result = env_mgr.reset(seed=seed)
        
        # Metrics tracking
        episode_loss = 0.0
        episode_td_error = 0.0
        episode_q_val = 0.0
        update_count = 0

        start_time = time.time()

        for step in range(max_steps):
            # 1. Select action
            action = agent.act(state=result.raw_state, discrete_state=result.discrete_state)

            # 2. Step environment
            prev_result = result
            result = env_mgr.step(action)
            
            # 3. Enforce max steps if env didn't terminate
            if step >= max_steps - 1 and not result.terminated:
                result.truncated = True

            # 4. Create transition
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

            # 5. Agent update
            metrics = agent.update(transition)
            
            # Aggregate metrics
            if metrics:
                update_count += 1
                if "loss" in metrics:
                    episode_loss += metrics["loss"]
                if "td_error" in metrics:
                    episode_td_error += metrics["td_error"]
                if "avg_q" in metrics:
                    episode_q_val += metrics["avg_q"]

            # Callback for Live Tracking UI
            if step_callback:
                step_callback(result, metrics)

            if result.terminated or result.truncated:
                break

        duration = time.time() - start_time

        # Compile final metrics summary
        summary = {
            "vehicle": vehicle,
            "steps": env_mgr.step_count,
            "total_reward": env_mgr.total_reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "duration": duration,
        }
        
        if update_count > 0:
            if vehicle == "R":
                summary["avg_loss"] = episode_loss / update_count
                summary["epsilon"] = self.dqn.epsilon
            else:
                summary["avg_td_error"] = episode_td_error / update_count
                summary["epsilon"] = self.sarsa.epsilon
            summary["avg_q"] = episode_q_val / update_count

        return summary

    def train_step(self, vehicle: str, env_mgr: EnvManager) -> dict[str, Any]:
        """
        Run exactly one training step. Useful for live UI tracking.
        Must be called after env_mgr.reset().
        """
        agent = self.dqn if vehicle == "R" else self.sarsa
        agent.set_eval_mode(False)

        # We assume prev_result is stored somewhere, or we can just fetch it from env_mgr
        # Wait, env_mgr doesn't store the current state natively, it just returns StepResult.
        # But EnvManager *does* track step_count.
        # Let's assume the UI passes the last StepResult, OR we can just get the raw observation from the env directly?
        # Better: we just run one step and return the result.
        # To do a proper RL update we need the PREVIOUS state.
        pass # Will rewrite this carefully

    def evaluate_episode(
        self,
        vehicle: str,
        env_mgr: EnvManager,
        seed: Optional[int] = None,
        max_steps: int = 1000,
        step_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Run one autonomous evaluation episode (greedy policy, no learning).
        """
        if vehicle not in ("R", "S"):
            raise ValueError(f"Vehicle must be 'R' or 'S', got {vehicle!r}")

        agent = self.dqn if vehicle == "R" else self.sarsa
        agent.set_eval_mode(True)

        result = env_mgr.reset(seed=seed)
        start_time = time.time()

        for step in range(max_steps):
            action = agent.act(state=result.raw_state, discrete_state=result.discrete_state)
            result = env_mgr.step(action)
            
            if step >= max_steps - 1 and not result.terminated:
                result.truncated = True

            if step_callback:
                step_callback(result, {})

            if result.terminated or result.truncated:
                break

        duration = time.time() - start_time

        return {
            "vehicle": vehicle,
            "steps": env_mgr.step_count,
            "total_reward": env_mgr.total_reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "duration": duration,
        }

    def save_checkpoints(self) -> None:
        """Save both agents to the configured checkpoint directory."""
        logger.info("Saving checkpoints to %s", self.checkpoint_dir)
        self.dqn.save(self.checkpoint_dir)
        self.sarsa.save(self.checkpoint_dir)

    def load_checkpoints(self) -> None:
        """Load both agents from the configured checkpoint directory."""
        logger.info("Loading checkpoints from %s", self.checkpoint_dir)
        self.dqn.load(self.checkpoint_dir)
        self.sarsa.load(self.checkpoint_dir)
