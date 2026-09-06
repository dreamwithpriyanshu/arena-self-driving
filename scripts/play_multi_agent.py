"""
Standalone PyGame script for real-time Multi-Agent Evaluation & Training.
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
from src.data.schemas import Transition


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Evaluation & Live Training Interface")
    parser.add_argument("--vehicles-count", type=int, default=20, help="Number of NPC vehicles on the road")
    parser.add_argument("--duration", type=int, default=120, help="Max duration of the episode in steps")
    parser.add_argument("--vehicles-density", type=float, default=1.0, help="Traffic density multiplier")
    parser.add_argument("--train", action="store_true", help="Enable live training/exploration during gameplay")
    parser.add_argument("--save", action="store_true", help="Save agent checkpoints after running (useful with --train)")
    
    args = parser.parse_args()

    pygame.init()

    # ── Pre-game instruction screen ──────────────────────────────
    screen = pygame.display.set_mode((640, 300))
    mode_text = "TRAINING & EVALUATION" if args.train else "EVALUATION"
    pygame.display.set_caption(f"Multi-Agent {mode_text} (DQN + SARSA)")
    font_big = pygame.font.SysFont("consolas", 24, bold=True)
    font = pygame.font.SysFont("consolas", 18)

    lines = [
        f"MULTI-AGENT {mode_text}: DQN vs SARSA",
        "",
        "Both agents will drive autonomously in the SAME simulation.",
        f"Live Training: {'ENABLED (Exploring)' if args.train else 'DISABLED (Greedy)'}",
        "",
        "  ESC  — Quit at any time",
        "",
        "Press ENTER to start...",
    ]

    screen.fill((30, 30, 36))
    for i, line in enumerate(lines):
        f = font_big if i == 0 else font
        color = (0, 255, 128) if i == 0 else (200, 200, 200)
        if "ENABLED" in line:
            color = (0, 255, 100)
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

    agent_dqn.set_eval_mode(not args.train)
    agent_sarsa.set_eval_mode(not args.train)

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
    
    update_count = [0, 0]
    total_loss = [0.0, 0.0]

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

            # HighwayEnv may return per-agent observations as 1-D ego-feature vectors
            # (shape (F,)) instead of the full (V, F) kinematics matrix. Normalize
            # both cases into a (V, F) matrix by padding absent neighbour rows with zeros.
            def _ensure_2d_matrix(obs_arr: np.ndarray) -> np.ndarray:
                obs_arr = np.asarray(obs_arr, dtype=np.float32)
                if obs_arr.ndim == 2:
                    return obs_arr
                # obs_arr is 1-D: infer features and expected vehicles_count from env config
                features = obs_arr.size
                try:
                    cfg = getattr(env.unwrapped, "config", {}) or {}
                    vehicles_expected = int(cfg.get("observation", {}).get("vehicles_count", cfg.get("vehicles_count", 6)))
                except Exception:
                    vehicles_expected = 6
                mat = np.zeros((vehicles_expected, features), dtype=np.float32)
                mat[0, :features] = obs_arr
                return mat

            obs_dqn = _ensure_2d_matrix(obs_dqn)
            obs_sarsa = _ensure_2d_matrix(obs_sarsa)

            state_dqn = build_raw_state(obs_dqn)
            discrete_state_dqn = build_discrete_state(obs_dqn, lanes_count=env.unwrapped.config.get("lanes_count", 4))
            
            state_sarsa = build_raw_state(obs_sarsa)
            discrete_state_sarsa = build_discrete_state(obs_sarsa, lanes_count=env.unwrapped.config.get("lanes_count", 4))

            # Get actions
            action_dqn = agent_dqn.act(state=state_dqn, discrete_state=discrete_state_dqn)
            action_sarsa = agent_sarsa.act(state=state_sarsa, discrete_state=discrete_state_sarsa)

            # Step environment (expects tuple of actions)
            next_obs_tuple, rewards, terminated_tuple, truncated_tuple, info = env.step((action_dqn, action_sarsa))
            
            total_reward[0] += rewards[0]
            total_reward[1] += rewards[1]
            step_count += 1
            
            # Record transitions and update
            if args.train:
                next_obs_dqn = np.asarray(next_obs_tuple[0], dtype=np.float32)
                next_obs_sarsa = np.asarray(next_obs_tuple[1], dtype=np.float32)

                # Ensure 2-D format for next observations as well
                next_obs_dqn = _ensure_2d_matrix(next_obs_dqn)
                next_obs_sarsa = _ensure_2d_matrix(next_obs_sarsa)
                
                # DQN Update
                t_dqn = Transition(
                    step=step, state=state_dqn.tolist(), action=action_dqn, reward=rewards[0],
                    next_state=build_raw_state(next_obs_dqn).tolist(), terminated=terminated_tuple[0], truncated=truncated_tuple[0],
                    discrete_state=discrete_state_dqn, next_discrete_state=build_discrete_state(next_obs_dqn, lanes_count=env.unwrapped.config.get("lanes_count", 4)),
                    lane=0, speed=0.0 # Multi-agent ignores these for training
                )
                m_dqn = agent_dqn.update(t_dqn)
                if m_dqn and "loss" in m_dqn:
                    update_count[0] += 1
                    total_loss[0] += m_dqn["loss"]
                    
                # SARSA Update
                t_sarsa = Transition(
                    step=step, state=state_sarsa.tolist(), action=action_sarsa, reward=rewards[1],
                    next_state=build_raw_state(next_obs_sarsa).tolist(), terminated=terminated_tuple[1], truncated=truncated_tuple[1],
                    discrete_state=discrete_state_sarsa, next_discrete_state=build_discrete_state(next_obs_sarsa, lanes_count=env.unwrapped.config["lanes_count"]),
                    lane=0, speed=0.0 # Multi-agent ignores these for training
                )
                m_sarsa = agent_sarsa.update(t_sarsa)
                if m_sarsa and "td_error" in m_sarsa:
                    update_count[1] += 1
                    total_loss[1] += m_sarsa["td_error"]

            obs_tuple = next_obs_tuple
            clock.tick(60)

            # If either agent crashes, they both stop
            if any(terminated_tuple) or any(truncated_tuple) or step >= args.duration - 1:
                break

        print(f"\nEpisode Ended!")
        print(f"Steps: {step_count}")
        print(f"DQN Total Reward: {total_reward[0]:.2f} (Crashed: {terminated_tuple[0]})")
        print(f"SARSA Total Reward: {total_reward[1]:.2f} (Crashed: {terminated_tuple[1]})")
        
        if args.train:
            print(f"DQN Updates: {update_count[0]} | Avg Loss: {(total_loss[0]/max(1, update_count[0])):.4f}")
            print(f"SARSA Updates: {update_count[1]} | Avg TD: {(total_loss[1]/max(1, update_count[1])):.4f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        pygame.quit()
        
        if args.save:
            agent_dqn.save(checkpoint_dir)
            agent_sarsa.save(checkpoint_dir)
            print("✅ Checkpoints saved successfully.")


if __name__ == "__main__":
    main()
