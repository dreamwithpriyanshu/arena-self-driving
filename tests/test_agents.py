"""Tests for the project's single tabular SARSA policy."""

from __future__ import annotations

from pathlib import Path

from src.agents.sarsa import SARSAAgent
from src.data.schemas import Transition


def _transition() -> Transition:
    return Transition(
        step=0, state=[0.0] * 36, action=1, reward=0.5,
        next_state=[0.1] * 36, terminated=False, truncated=False,
        discrete_state=10, next_discrete_state=11, lane=0, speed=25.0,
    )


def test_sarsa_selects_and_updates_with_the_selected_next_action() -> None:
    agent = SARSAAgent(epsilon_start=0.0)
    agent.q_table[11, 3] = 2.0

    action = agent.act(state=None, discrete_state=10)
    assert 0 <= action < agent.num_actions

    metrics = agent.update(_transition(), next_action=3)
    assert metrics["td_error"] > 0
    assert agent.q_table[10, 1] > 0


def test_sarsa_checkpoint_round_trip() -> None:
    import shutil

    ckpt = Path("artifacts/test_checkpoints_agents")
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    try:
        agent = SARSAAgent()
        agent.update(_transition(), next_action=1)
        agent.save(ckpt)

        loaded = SARSAAgent()
        loaded.load(ckpt)
        assert loaded.q_table.shape == agent.q_table.shape
        assert loaded.q_table[10, 1] == agent.q_table[10, 1]
    finally:
        shutil.rmtree(ckpt, ignore_errors=True)
