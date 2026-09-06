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
from src.envs.actions import action_name
from src.data.schemas import Transition


def _vehicle_reward(env, vehicle, action: int) -> float:
    """Calculate the HighwayEnv reward terms for one controlled vehicle."""
    cfg = env.unwrapped.config
    neighbours = env.unwrapped.road.network.all_side_lanes(vehicle.lane_index)
    lane = (
        vehicle.target_lane_index[2]
        if hasattr(vehicle, "target_lane_index")
        else vehicle.lane_index[2]
    )
    speed_min, speed_max = cfg.get("reward_speed_range", [20, 30])
    forward_speed = float(vehicle.speed * np.cos(vehicle.heading))
    speed_fraction = float(np.clip((forward_speed - speed_min) / max(speed_max - speed_min, 1e-6), 0, 1))
    rewards = {
        "collision_reward": float(vehicle.crashed),
        "right_lane_reward": lane / max(len(neighbours) - 1, 1),
        "high_speed_reward": speed_fraction,
        "on_road_reward": float(vehicle.on_road),
    }
    value = sum(cfg.get(name, 0) * term for name, term in rewards.items())
    if cfg.get("normalize_reward", True):
        value = (value - cfg.get("collision_reward", -1.0)) / max(
            cfg.get("high_speed_reward", 0.4) + cfg.get("right_lane_reward", 0.1)
            - cfg.get("collision_reward", -1.0),
            1e-6,
        )
    return float(value * rewards["on_road_reward"])


def _prepare_visible_agents(env) -> None:
    """Place R and S beside each other so both remain in the camera view."""
    road = env.unwrapped.road
    vehicles = list(getattr(env.unwrapped, "controlled_vehicles", []))
    if len(vehicles) < 2:
        return

    from_node, to_node = next(iter(road.network.graph.keys())), None
    to_node = next(iter(road.network.graph[from_node].keys()))
    lane_count = int(env.unwrapped.config.get("lanes_count", 4))
    anchor_lane = road.network.get_lane((from_node, to_node, 1))
    anchor_x = anchor_lane.local_coordinates(vehicles[0].position)[0]
    anchor_x = float(np.clip(anchor_x, 60.0, max(60.0, anchor_lane.length - 60.0)))

    for index, (vehicle, colour) in enumerate(
        zip(vehicles[:2], ((0, 229, 255), (255, 145, 0)))
    ):
        lane_id = min(index + 1, lane_count - 1)
        lane_index = (from_node, to_node, lane_id)
        lane = road.network.get_lane(lane_index)
        vehicle.position = lane.position(anchor_x, 0)
        vehicle.heading = lane.heading_at(anchor_x)
        vehicle.speed = 22.0
        vehicle.target_speed = 22.0
        vehicle.target_lane_index = lane_index
        vehicle.color = colour
        vehicle.on_state_update()


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
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 6,
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "normalize": True,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"},
        },
    }

    env = create_highway_env(render_mode="human", config_overrides=config_overrides)
    obs_tuple, info = env.reset()
    _prepare_visible_agents(env)
    # Re-read observations after repositioning the controlled vehicles.
    obs_tuple = env.unwrapped.observation_type.observe()
    
    clock = pygame.time.Clock()
    step_count = 0
    total_reward = [0.0, 0.0]
    
    update_count = [0, 0]
    total_loss = [0.0, 0.0]

    # The HighwayEnv road is 600x300 by default. The HUD is drawn over the
    # native PyGame surface after each environment step.
    hud_font = pygame.font.SysFont("consolas", 16)
    hud_small = pygame.font.SysFont("consolas", 13)
    paused = False
    show_hud = True

    def draw_hud(actions, vehicles, terminated, truncated):
        """Draw live controls and per-agent telemetry on the native window."""
        if not show_hud:
            return
        surface = pygame.display.get_surface()
        if surface is None:
            return
        width, height = surface.get_size()
        panel_height = 112
        panel = pygame.Surface((width, panel_height), pygame.SRCALPHA)
        panel.fill((22, 22, 28, 228))
        surface.blit(panel, (0, height - panel_height))

        title = "MULTI-AGENT LIVE DATA"
        if paused:
            title += "  |  PAUSED"
        surface.blit(hud_font.render(title, True, (0, 229, 255)), (12, height - 105))
        surface.blit(
            hud_small.render(
                "SPACE pause/resume   H HUD   S save checkpoints   ESC quit",
                True,
                (210, 210, 215),
            ),
            (12, height - 84),
        )
        rows = []
        for index, (label, colour) in enumerate((("R / DQN", (0, 229, 255)), ("S / SARSA", (255, 145, 0)))):
            vehicle = vehicles[index] if index < len(vehicles) else None
            lane = "-"
            speed = "-"
            crashed = bool(terminated[index]) if index < len(terminated) else False
            if vehicle is not None:
                lane_index = getattr(vehicle, "lane_index", "-")
                lane = lane_index[-1] if isinstance(lane_index, (tuple, list)) else lane_index
                speed = f"{float(getattr(vehicle, 'speed', 0.0)):.1f}"
            status = "CRASHED" if crashed else ("TIMEOUT" if truncated[index] else "RUNNING")
            rows.append(
                f"{label:<9} action={action_name(int(actions[index])):<11} "
                f"reward={total_reward[index]:>7.2f} lane={lane} speed={speed:>5} m/s {status}"
            )
            surface.blit(hud_small.render(rows[-1], True, colour), (12, height - 62 + index * 18))

    try:
        for step in range(args.duration):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key == pygame.K_s:
                        agent_dqn.save(checkpoint_dir)
                        agent_sarsa.save(checkpoint_dir)
                        print("Checkpoints saved from GUI control.")
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

            if paused:
                draw_hud((0, 0), getattr(env.unwrapped, "controlled_vehicles", []), (False, False), (False, False))
                pygame.display.flip()
                clock.tick(30)
                continue

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
                # If a 1-D ego vector is provided, prefer reconstructing the full
                # kinematics matrix from the environment state (best-effort). Fall
                # back to simple zero-padding if reconstruction isn't available.
                try:
                    from src.envs.state_builder import build_raw_state_from_env
                    reconstructed = build_raw_state_from_env(env)
                    if reconstructed is not None:
                        return reconstructed
                except Exception:
                    pass

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

            # MultiAgentAction accepts one action per controlled vehicle.
            next_obs_tuple, rewards, terminated, truncated, info = env.step((action_dqn, action_sarsa))

            # HighwayEnv's HighwayEnv reward/termination contract is scalar,
            # even with two controlled vehicles. Derive per-vehicle status and
            # rewards from each controlled vehicle for accurate live telemetry.
            controlled = getattr(env.unwrapped, "controlled_vehicles", [])
            terminated_tuple = tuple(
                bool(getattr(vehicle, "crashed", False)) for vehicle in controlled[:2]
            )
            terminated_tuple = terminated_tuple or (bool(terminated), bool(terminated))
            if len(terminated_tuple) < 2:
                terminated_tuple = tuple(terminated_tuple) + (bool(terminated),) * (2 - len(terminated_tuple))
            truncated_tuple = (bool(truncated), bool(truncated))
            rewards = tuple(
                _vehicle_reward(env, vehicle, action)
                for vehicle, action in zip(controlled[:2], (action_dqn, action_sarsa))
            )
            if len(rewards) < 2:
                rewards = rewards + (float(rewards[0]) if rewards else float(0.0),) * (2 - len(rewards))
            
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
            draw_hud((action_dqn, action_sarsa), controlled, terminated_tuple, truncated_tuple)
            pygame.display.flip()
            clock.tick(60)

            # If either agent crashes, stop stepping and show the result HUD.
            if any(terminated_tuple) or any(truncated_tuple) or step >= args.duration - 1:
                break

        print(f"\nEpisode Ended!")
        print(f"Steps: {step_count}")
        print(f"DQN Total Reward: {total_reward[0]:.2f} (Crashed: {terminated_tuple[0]})")
        print(f"SARSA Total Reward: {total_reward[1]:.2f} (Crashed: {terminated_tuple[1]})")
        
        if args.train:
            print(f"DQN Updates: {update_count[0]} | Avg Loss: {(total_loss[0]/max(1, update_count[0])):.4f}")
            print(f"SARSA Updates: {update_count[1]} | Avg TD: {(total_loss[1]/max(1, update_count[1])):.4f}")

        # Keep the native window open after a crash/timeout. The user decides
        # when to close it or save the updated checkpoints.
        finished = True
        while finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    finished = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    agent_dqn.save(checkpoint_dir)
                    agent_sarsa.save(checkpoint_dir)
                    print("Checkpoints saved from GUI control.")
            draw_hud(
                (action_dqn, action_sarsa),
                getattr(env.unwrapped, "controlled_vehicles", []),
                terminated_tuple,
                truncated_tuple,
            )
            surface = pygame.display.get_surface()
            if surface is not None:
                message = hud_font.render(
                    "EPISODE COMPLETE — S saves checkpoints   ESC closes",
                    True,
                    (255, 255, 255),
                )
                surface.blit(message, (12, 8))
                pygame.display.flip()
            clock.tick(30)

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
