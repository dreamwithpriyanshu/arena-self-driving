"""
src.training — Training Orchestration.

Manages the end-to-end autonomous training and evaluation loops, 
tying together agents, the environment manager, and data loaders.
"""

from src.training.orchestrator import TrainingOrchestrator

__all__ = ["TrainingOrchestrator"]
