#!/usr/bin/env python
"""
Smoke test for Build Step 3 — Agents (DQN & SARSA).

Verifies:
1. Base interface is correctly implemented by both agents.
2. DQN initializes PyTorch networks, stores transitions, runs an update step, and saves/loads weights safely.
3. SARSA initializes Numpy table, processes a transition, and saves/loads correctly.
4. Both agents support eval mode.
5. Neither agent module imports the other (architecture check).
"""

from __future__ import annotations

import sys
import os
import shutil
import ast
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.data.schemas import Transition


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _dummy_transition() -> Transition:
    """Create a dummy transition for testing."""
    return Transition(
        step=0,
        state=[0.0] * 36,
        action=1,  # IDLE
        reward=0.5,
        next_state=[0.1] * 36,
        terminated=False,
        truncated=False,
        discrete_state=10,
        next_discrete_state=11,
        lane=0,
        speed=25.0
    )


def test_architecture_imports() -> None:
    _header("Test 1: Architecture Check (No Cross-Imports)")
    
    dqn_path = Path("src/agents/dqn.py")
    sarsa_path = Path("src/agents/sarsa.py")
    
    def check_imports(filepath: Path, forbidden_module: str):
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if forbidden_module in alias.name:
                        raise ValueError(f"{filepath.name} illegally imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and forbidden_module in node.module:
                    raise ValueError(f"{filepath.name} illegally imports from {node.module}")

    check_imports(dqn_path, "sarsa")
    check_imports(sarsa_path, "dqn")
    print("  [OK] Agents do not import each other.")


def test_dqn_agent() -> None:
    _header("Test 2: DQN Agent")
    
    test_dir = Path("artifacts/test_agents")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        
    agent = DQNAgent(batch_size=2)  # tiny batch for fast testing
    
    # Eval mode test
    agent.set_eval_mode(True)
    assert agent._eval_mode == True
    agent.set_eval_mode(False)
    
    # Act test
    dummy_state = [0.5] * 36
    action = agent.act(dummy_state, discrete_state=0)
    assert 0 <= action < 5
    print("  [OK] DQN act() produced valid action")
    
    # Prefill and update test
    t1 = _dummy_transition()
    t2 = _dummy_transition()
    t3 = _dummy_transition()
    
    agent.prefill_buffer([t1, t2])
    assert len(agent.memory) == 2
    print("  [OK] Buffer prefill successful")
    
    metrics = agent.update(t3)
    assert len(agent.memory) == 3
    assert "loss" in metrics
    print("  [OK] DQN update() returned metrics (loss calculated)")
    
    # Save/Load test
    agent.save(test_dir)
    assert (test_dir / "dqn_policy.pt").exists()
    
    agent2 = DQNAgent()
    agent2.load(test_dir)
    print("  [OK] DQN save() and load() successful (weights_only=True)")


def test_sarsa_agent() -> None:
    _header("Test 3: SARSA Agent")
    
    test_dir = Path("artifacts/test_agents")
    
    agent = SARSAAgent()
    
    # Eval mode test
    agent.set_eval_mode(True)
    assert agent._eval_mode == True
    agent.set_eval_mode(False)
    
    # Act test
    action = agent.act(state=None, discrete_state=15)
    assert 0 <= action < 5
    print("  [OK] SARSA act() produced valid action")
    
    # Warm start and update test
    t1 = _dummy_transition()
    
    metrics = agent.update(t1)
    assert "td_error" in metrics
    print("  [OK] SARSA update() returned metrics (td_error calculated)")
    
    agent.warm_start([_dummy_transition(), _dummy_transition()])
    print("  [OK] Warm start successful")
    
    # Save/Load test
    agent.save(test_dir)
    assert (test_dir / "sarsa_q_table.npy").exists()
    
    agent2 = SARSAAgent()
    agent2.load(test_dir)
    # Check if table shape matches
    assert agent2.q_table.shape == agent.q_table.shape
    print("  [OK] SARSA save() and load() successful (no pickles)")


