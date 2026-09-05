import gymnasium as gym
import highway_env
import numpy as np
from src.envs.highway_factory import create_highway_env

env = create_highway_env(config_overrides={"observation": {"normalize": False}}, render_mode="rgb_array")
obs, info = env.reset()

print("OBS SHAPE:", obs.shape)
print("EGO ROW (obs[0]):", obs[0])
for i in range(1, len(obs)):
    print(f"NPC {i}:", obs[i])
