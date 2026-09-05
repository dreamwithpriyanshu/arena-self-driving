"""
src.simulation — Simulation façade.

The ONLY module that Streamlit pages are allowed to import from ``src``.
"""

from src.simulation.env_manager import EnvManager, RenderData, StepResult

__all__ = ["EnvManager", "RenderData", "StepResult"]
