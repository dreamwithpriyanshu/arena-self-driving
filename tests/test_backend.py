"""Contract tests for the local FastAPI training control surface."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.main import TrainingRequest, app, training_command


def test_only_sarsa_model_is_accepted() -> None:
    assert TrainingRequest().agent_type == "S"
    with pytest.raises(ValidationError):
        TrainingRequest(agent_type="R")  # type: ignore[arg-type]


def test_training_command_is_fixed_argument_vector() -> None:
    command = training_command(TrainingRequest(episodes=3, seed=7), Path("artifacts/runs/test"))
    assert command[1].endswith("scripts\\train.py") or command[1].endswith("scripts/train.py")
    assert "--episodes" in command and "3" in command
    assert "--seed" in command and "7" in command
    assert "shell" not in " ".join(command)


def test_expected_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}
    assert {"/", "/training/start", "/training/status/{run_id}", "/training/stop/{run_id}", "/sessions/human/start", "/sessions/agent/start", "/demonstrations/summary", "/runs", "/documentation", "/documentation/{document_id}", "/ws/training/{run_id}"} <= paths
