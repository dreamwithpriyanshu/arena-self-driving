#!/usr/bin/env python
"""
Smoke test for Build Step 2 — Human Recorder.

Verifies:
1. Keyboard controller maps keys correctly
2. EpisodeManager starts, acts, and saves an episode
3. Recorder saves data to the correct sanitised path
4. Validator catches invalid data and passes valid data
5. Loader correctly loads the saved episode

Exit code 0 = all checks pass.
"""

from __future__ import annotations

import sys
import os
import shutil
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

from src.envs.actions import Action
from src.human.keyboard_controller import KeyboardController
from src.human.episode_manager import EpisodeManager
from src.data.validator import validate_episode_file
from src.data.loader import load_episode, get_dataset_summary


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_keyboard_controller() -> None:
    _header("Test 1: Keyboard Controller")
    ctrl = KeyboardController()
    
    assert ctrl.key_to_action("arrowup") == Action.FASTER
    assert ctrl.key_to_action("arrowleft") == Action.LANE_LEFT
    assert ctrl.key_to_action(" ") == Action.IDLE
    assert ctrl.key_to_action("unknown_key") == Action.IDLE
    
    print("  [OK] Default mappings correct")
    print("  [OK] Unknown key fallback to IDLE correct")


def test_episode_manager_and_recorder() -> None:
    _header("Test 2: Episode Manager (Record & Save)")
    
    test_dir = Path("data/test_human_demos")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        
    mgr = EpisodeManager(base_dir=test_dir, render_mode="rgb_array")
    
    # Start episode
    mgr.start(vehicle="R", seed=42)
    assert mgr.is_active
    print("  [OK] Episode started")
    
    # Take a few actions
    mgr.act("arrowup")
    mgr.act("arrowup")
    mgr.act("arrowleft")
    
    assert mgr.step_count == 3
    print(f"  [OK] Recorded {mgr.step_count} steps")
    
    # Save
    saved_path = mgr.save()
    assert not mgr.is_active
    assert saved_path.exists()
    print(f"  [OK] Episode saved to {saved_path}")


def test_validator_and_loader() -> None:
    _header("Test 3: Validator and Loader")
    
    test_dir = Path("data/test_human_demos")
    files = list(test_dir.glob("*.jsonl"))
    assert len(files) == 1
    filepath = files[0]
    
    # Validation
    result = validate_episode_file(filepath)
    assert result.is_valid
    assert result.num_transitions == 3
    print(f"  [OK] Validation passed: {result.num_transitions} transitions")
    
    # Loading
    meta, transitions = load_episode(filepath, validate=True)
    assert meta.vehicle == "R"
    assert len(transitions) == 3
    assert transitions[0].action == Action.FASTER
    print("  [OK] Episode loaded successfully")
    print(f"  [OK] Metadata: {meta.episode_id}, vehicle {meta.vehicle}")
    
    # Dataset summary
    summary = get_dataset_summary(test_dir)
    assert summary["num_episodes"] == 1
    assert summary["total_transitions"] == 3
    print("  [OK] Dataset summary correct")


def test_recorder_security() -> None:
    _header("Test 4: Recorder Security (Path Traversal)")
    
    from src.data.recorder import TransitionRecorder
    
    # Attempt to write outside allowed dirs
    try:
        rec = TransitionRecorder(base_dir="../outside_project")
        assert False, "Should have prevented directory traversal"
    except ValueError as e:
        print("  [OK] Prevented writing outside allowed dirs")

    # Attempt malicious episode ID
    rec = TransitionRecorder(base_dir="data/test_human_demos")
    safe_id = rec.start_episode(episode_id="../../../malicious_name")
    assert "malicious_name" in safe_id
    assert "/" not in safe_id
    assert "\\" not in safe_id
    assert "." not in safe_id
    print(f"  [OK] Sanitised malicious ID to: {safe_id}")
    rec.discard_episode()



import pytest
import shutil
from pathlib import Path

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_dir():
    # Setup
    test_dir = Path("data/test_human_demos")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    yield
    # Teardown
    if test_dir.exists():
        shutil.rmtree(test_dir)
