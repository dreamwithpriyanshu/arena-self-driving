"""
src.simulation — Simulation Facade.

The sole entry point for Streamlit pages to interact with the environment,
training loops, human recording, and data.
"""

from src.simulation.env_manager import EnvManager
from src.simulation.training_facade import TrainingFacade
from src.simulation.human_facade import HumanFacade
from src.simulation.renderer import render_highway_svg

__all__ = ["EnvManager", "TrainingFacade", "HumanFacade", "render_highway_svg"]
