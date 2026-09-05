"""
src.simulation — Simulation Facade.

The sole entry point for Streamlit pages to interact with the environment,
training loops, human recording, and data.
"""

from src.simulation.env_manager import EnvManager
from src.simulation.training_facade import TrainingFacade
from src.simulation.human_facade import HumanFacade

__all__ = ["EnvManager", "TrainingFacade", "HumanFacade"]
