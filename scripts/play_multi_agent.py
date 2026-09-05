"""
Standalone PyGame script for real-time Multi-Agent Evaluation.
Watch BOTH the trained DQN (R) and SARSA (S) agents drive autonomously 
in the same environment!
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
import numpy as np

from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.envs.highway_factory import create_highway_env
from src.envs.state_builder import build_raw_state, build_discrete_state


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Evaluation Interface")
    parser.add_argument("--vehicles-count", type=int, default=20, help="Number of NPC vehicles on the road")
    parser.add_argument("--duration", type=int, default=120, help="Max duration of the episode in steps")
    parser.add_argument("--vehicles-density", type=float, default=1.0, help="Traffic density multiplier")
    
    args = parser.parse_args()

    pygame.init()

    # ── Pre-game instruction screen ──────────────────────────────
    screen = pygame.display.set_mode((640, 300))
    pygame.display.set_caption(f"Multi-Agent Evaluation (DQN + SARSA)")
    font_big = pygame.font.SysFont("consolas", 24, bold=True)
    font = pygame.font.SysFont("consolas", 18)

    lines = [
        "MULTI-AGENT EVALUATION: DQN vs SARSA",
        "",
        "Both agents will drive autonomously in the SAME simulation.",
        "They must navigate traffic and avoid each other.",
        "",
        "  ESC  — Quit at any time",
        "",
        "Press ENTER to start...",
    ]

    screen.fill((30, 30, 36))
    for i, line in enumerate(lines):
        f = font_big if i == 0 else font
        color = (0, 255, 128) if i == 0 else (200, 200, 200)
        surf = f.render(line, True, color)
        screen.blit(surf, (30, 30 + i * 32))
    pygame.display.flip()

    # Wait for ENTER
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

    # Close the instruction window
    pygame.display.quit()

    # ── Load agents ──────────────────────────────────────────────
    checkpoint_dir = Path("artifacts/checkpoints")

    agent_dqn = DQNAgent()
    try:
        agent_dqn.load(checkpoint_dir)
        print("✅ DQN checkpoint loaded.")
    except Exception as e:
        print(f"⚠️  No DQN checkpoint found. Acting randomly.")

    agent_sarsa = SARSAAgent()
    try:
        agent_sarsa.load(checkpoint_dir)
        print("✅ SARSA checkpoint loaded.")
    except Exception as e:
        print(f"⚠️  No SARSA checkpoint found. Acting randomly.")

    agent_dqn.set_eval_mode(True)
    agent_sarsa.set_eval_mode(True)

    # ── Run evaluation episode ───────────────────────────────────
    config_overrides = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
        "vehicles_density": args.vehicles_density,
        "controlled_vehicles": 2, # Two controlled agents!
    }

    env = create_highway_env(render_mode="human", config_overrides=config_overrides)
    obs_tuple, info = env.reset()
    
    clock = pygame.time.Clock()
    step_count = 0
    total_reward = [0.0, 0.0]

    try:
        for step in range(args.duration):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

            # Extract observations for each agent
            obs_dqn = np.asarray(obs_tuple[0], dtype=np.float32)
            obs_sarsa = np.asarray(obs_tuple[1], dtype=np.float32)

            # Get actions
            action_dqn = agent_dqn.act(
                state=build_raw_state(obs_dqn),
                discrete_state=build_discrete_state(obs_dqn, lanes_count=env.unwrapped.config["lanes_count"])
            )
            action_sarsa = agent_sarsa.act(
                state=build_raw_state(obs_sarsa),
                discrete_state=build_discrete_state(obs_sarsa, lanes_count=env.unwrapped.config["lanes_count"])
            )

            # Step environment (expects tuple of actions)
            obs_tuple, rewards, terminated_tuple, truncated_tuple, info = env.step((action_dqn, action_sarsa))
            
            total_reward[0] += rewards[0]
            total_reward[1] += rewards[1]
            step_count += 1

            clock.tick(5)

            # If either agent crashes, they both stop
            if any(terminated_tuple) or any(truncated_tuple):
                break

        print(f"\nEpisode Ended!")
        print(f"Steps: {step_count}")
        print(f"DQN Total Reward: {total_reward[0]:.2f} (Crashed: {terminated_tuple[0]})")
        print(f"SARSA Total Reward: {total_reward[1]:.2f} (Crashed: {terminated_tuple[1]})")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
