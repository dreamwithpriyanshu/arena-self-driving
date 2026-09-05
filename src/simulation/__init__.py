"""
src.simulation — Simulation Facade.

The sole entry point for Streamlit pages to interact with the environment,
training loops, human recording, and data.

Uses lazy imports to avoid circular dependency with src.training.
"""

from src.simulation.env_manager import EnvManager
from src.simulation.human_facade import HumanFacade
from src.simulation.renderer import render_highway_svg


def __getattr__(name):
    """Lazy import TrainingFacade to break circular dependency."""
    if name == "TrainingFacade":
        from src.simulation.training_facade import TrainingFacade
        return TrainingFacade
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EnvManager", "HumanFacade", "render_highway_svg"]
