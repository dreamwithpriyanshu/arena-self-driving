#!/usr/bin/env python
"""
Smoke test for Build Step 1 — Environment, Actions and State.

Verifies:
1. Environment creation from YAML config
2. Reset produces a valid observation
3. 10 random steps work without errors
4. Raw recorded state has correct shape
5. Discrete state (SARSA) is within expected range
6. Action validation works
7. Environment closes cleanly

Exit code 0 = all checks pass.
"""

from __future__ import annotations

import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.envs.highway_factory import create_highway_env
from src.envs.actions import (
    NUM_ACTIONS,
    Action,
    action_name,
    is_valid_action,
    validate_action,
)
from src.envs.state_builder import (
    build_raw_state,
    build_discrete_state_with_lane,
    raw_state_dim,
    total_discrete_states,
)
from src.simulation.env_manager import EnvManager


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_actions() -> None:
    """Verify action constants and validation."""
    _header("Test 1: Action definitions")

    assert NUM_ACTIONS == 5, f"Expected 5 actions, got {NUM_ACTIONS}"
    print(f"  [OK] NUM_ACTIONS = {NUM_ACTIONS}")

    for a in Action:
        assert is_valid_action(a.value)
        name = action_name(a.value)
        print(f"  [OK] Action {a.value} = {name}")

    # Invalid action should be rejected
    assert not is_valid_action(-1)
    assert not is_valid_action(5)
    assert not is_valid_action(999)
    try:
        validate_action(99)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [OK] validate_action(99) correctly raised ValueError")

    print("  [OK] All action tests passed")


def test_env_create_and_reset() -> None:
    """Create env, reset, inspect observation."""
    _header("Test 2: Environment creation and reset")

    mgr = EnvManager(render_mode="rgb_array")
    result = mgr.reset(seed=42)

    print(f"  [OK] Environment created and reset (seed=42)")
    print(f"  [OK] raw_obs shape: {result.raw_obs.shape}")
    print(f"  [OK] raw_state shape: {result.raw_state.shape}")
    print(f"  [OK] discrete_state: {result.discrete_state}")
    print(f"  [OK] lane_index: {result.lane_index}")
    print(f"  [OK] speed: {result.speed:.2f}")

    # Shape checks
    expected_obs_shape = (6, 6)  # 6 vehicles x 6 features
    assert result.raw_obs.shape == expected_obs_shape, (
        f"Expected obs shape {expected_obs_shape}, got {result.raw_obs.shape}"
    )

    expected_state_dim = raw_state_dim()  # 36
    assert result.raw_state.shape == (expected_state_dim,), (
        f"Expected raw_state dim {expected_state_dim}, "
        f"got {result.raw_state.shape}"
    )

    # Discrete state range
    max_states = total_discrete_states()
    assert 0 <= result.discrete_state < max_states, (
        f"Discrete state {result.discrete_state} out of range [0, {max_states})"
    )
    print(f"  [OK] Discrete state in range [0, {max_states})")

    mgr.close()
    print("  [OK] Environment closed")


def test_stepping() -> None:
    """Run 10 random steps and verify outputs."""
    _header("Test 3: Stepping (10 random actions)")

    mgr = EnvManager(render_mode="rgb_array")
    result = mgr.reset(seed=42)

    rng = np.random.default_rng(123)
    max_states = total_discrete_states()

    for i in range(10):
        action = int(rng.integers(0, NUM_ACTIONS))
        result = mgr.step(action)

        # Verify types
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)
        assert 0 <= result.discrete_state < max_states

        status = "DONE" if (result.terminated or result.truncated) else "OK"
        print(
            f"  Step {i+1:2d}: action={action_name(action):12s}  "
            f"reward={result.reward:+.3f}  "
            f"discrete_state={result.discrete_state:4d}  "
            f"lane={result.lane_index}  "
            f"speed={result.speed:5.1f}  "
            f"[{status}]"
        )

        if result.terminated or result.truncated:
            print(f"  >> Episode ended at step {i+1}")
            break

    print(f"  [OK] Total reward: {mgr.total_reward:+.3f}")
    print(f"  [OK] Steps taken: {mgr.step_count}")

    mgr.close()
    print("  [OK] Environment closed cleanly")


def test_multi_agent_stepping() -> None:
    """Verify the native multi-agent action/observation contract."""
    _header("Test 4: Multi-agent tuple actions")

    env = create_highway_env(
        render_mode="rgb_array",
        config_overrides={
            "controlled_vehicles": 2,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 6,
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": False,
                    "normalize": True,
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"},
            },
        },
    )
    try:
        observations, _info = env.reset(seed=7)
        assert len(observations) == 2
        next_observations, reward, terminated, truncated, _info = env.step((1, 1))
        assert len(next_observations) == 2
        assert np.isscalar(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    finally:
        env.close()

    print("  [OK] Tuple action reached HighwayEnv with two controlled vehicles")



def test_state_builder_edge_cases() -> None:
    """Test state builder with edge-case observations."""
    _header("Test 5: State builder edge cases")

    # All-zero observation (no neighbours visible)
    obs_empty = np.zeros((6, 6), dtype=np.float32)
    raw = build_raw_state(obs_empty)
    assert raw.shape == (36,)
    print(f"  [OK] All-zero obs -> raw shape {raw.shape}")

    discrete = build_discrete_state_with_lane(obs_empty, lane_index=0)
    assert 0 <= discrete < total_discrete_states()
    print(f"  [OK] All-zero obs -> discrete state {discrete}")

    # Observation with 1-D ego-only vector should now be accepted and expanded
    raw = build_raw_state(np.zeros((10,)))
    expected_dim = raw_state_dim(vehicles_count=6, features_count=10)
    assert raw.shape == (expected_dim,)
    print(f"  [OK] 1-D obs expanded -> raw shape {raw.shape}")

    print("  [OK] Edge case tests passed")


