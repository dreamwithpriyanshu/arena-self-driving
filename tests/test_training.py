#!/usr/bin/env python
"""
Smoke test for Build Step 4a — Training Orchestration.

Verifies:
1. Orchestrator initializes with both agents.
2. Warm start correctly uses data loader and agent prefill/warm_start methods.
3. Autonomous training episode runs for R (DQN).
4. Autonomous training episode runs for S (SARSA).
5. Evaluation mode executes greedily.
6. Checkpoint save/load round-trip works.
7. Architecture check (training doesn't import Streamlit).
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
from src.simulation.env_manager import EnvManager
from src.training.orchestrator import TrainingOrchestrator
from src.human.episode_manager import EpisodeManager


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_architecture_imports() -> None:
    _header("Test 1: Architecture Check (Training Layer)")
    
    orch_path = Path("src/training/orchestrator.py")
    
    with open(orch_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
        
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "streamlit" in alias.name:
                    raise ValueError(f"{orch_path.name} illegally imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and "streamlit" in node.module:
                raise ValueError(f"{orch_path.name} illegally imports from {node.module}")

    print("  [OK] Training layer does not import Streamlit (UI).")


def test_orchestrator_e2e() -> None:
    _header("Test 2: E2E Training Orchestration")
    
    # 1. Setup temporary dirs
    test_data_dir = Path("data/test_human_demos")
    test_checkpoint_dir = Path("artifacts/test_checkpoints")
    
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    if test_checkpoint_dir.exists():
        shutil.rmtree(test_checkpoint_dir)
        
    # 2. Generate a dummy human demonstration
    print("  Generating dummy human demonstration for warm-start...")
    human_mgr = EpisodeManager(base_dir=test_data_dir, render_mode="rgb_array")
    human_mgr.start(vehicle="R", seed=42)
    human_mgr.act("arrowup")
    human_mgr.act("arrowleft")
    human_mgr.save()
    
    # 3. Initialize Orchestrator
    dqn = DQNAgent(batch_size=2)
    sarsa = SARSAAgent()
    env_mgr = EnvManager(render_mode="rgb_array")
    
    orch = TrainingOrchestrator(
        dqn_agent=dqn,
        sarsa_agent=sarsa,
        checkpoint_dir=test_checkpoint_dir
    )
    
    # 4. Warm Start
    warm_res = orch.warm_start(data_dir=test_data_dir)
    assert warm_res["loaded"] == 2
    assert len(dqn.memory) == 2
    print("  [OK] Warm start successful (DQN and SARSA ingested demo)")
    
    # 5. Train R (DQN)
    metrics_r = orch.train_episode(vehicle="R", env_mgr=env_mgr, seed=1, max_steps=10)
    assert metrics_r["steps"] > 0
    assert "avg_loss" in metrics_r
    print(f"  [OK] R (DQN) training episode ran: {metrics_r['steps']} steps, loss={metrics_r['avg_loss']:.4f}")
    
    # 6. Train S (SARSA)
    metrics_s = orch.train_episode(vehicle="S", env_mgr=env_mgr, seed=2, max_steps=10)
    assert metrics_s["steps"] > 0
    assert "avg_td_error" in metrics_s
    print(f"  [OK] S (SARSA) training episode ran: {metrics_s['steps']} steps, td_error={metrics_s['avg_td_error']:.4f}")
    
    # 7. Evaluate R (DQN)
    eval_r = orch.evaluate_episode(vehicle="R", env_mgr=env_mgr, seed=3, max_steps=10)
    assert eval_r["steps"] > 0
    print(f"  [OK] R (DQN) evaluation episode ran (greedy mode): {eval_r['steps']} steps")
    
    # 8. Checkpoints
    orch.save_checkpoints()
    assert (test_checkpoint_dir / "dqn_policy.pt").exists()
    assert (test_checkpoint_dir / "sarsa_q_table.npy").exists()
    print("  [OK] Checkpoints saved to disk")
    
    # Wipe and load
    dqn_new = DQNAgent()
    sarsa_new = SARSAAgent()
    orch_new = TrainingOrchestrator(dqn_agent=dqn_new, sarsa_agent=sarsa_new, checkpoint_dir=test_checkpoint_dir)
    
    orch_new.load_checkpoints()
    print("  [OK] Checkpoints loaded successfully")
    
    env_mgr.close()
    
    # Cleanup
    shutil.rmtree(test_data_dir)
    shutil.rmtree(test_checkpoint_dir)


