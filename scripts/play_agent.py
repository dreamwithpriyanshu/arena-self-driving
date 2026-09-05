"""
Standalone PyGame script for real-time Agent Evaluation & Training.
Watch the trained DQN or SARSA agent drive autonomously at 60 FPS!

Includes live training capabilities via the --train flag.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.simulation.env_manager import EnvManager
from src.data.schemas import Transition


def main():
    parser = argparse.ArgumentParser(description="Agent Evaluation & Live Training Interface")
    parser.add_argument("vehicle", type=str, choices=["R", "S"], help="Vehicle agent to watch: R (DQN) or S (SARSA)")
    parser.add_argument("--vehicles-count", type=int, default=15, help="Number of NPC vehicles on the road")
    parser.add_argument("--duration", type=int, default=120, help="Max duration of the episode in steps")
    parser.add_argument("--vehicles-density", type=float, default=1.0, help="Traffic density multiplier")
    parser.add_argument("--train", action="store_true", help="Enable live training/exploration during gameplay")
    parser.add_argument("--save", action="store_true", help="Save agent checkpoints after running (useful with --train)")
    
    args = parser.parse_args()
    vehicle = args.vehicle.upper()

    pygame.init()

    # ── Pre-game instruction screen ──────────────────────────────
    screen = pygame.display.set_mode((640, 300))
    mode_text = "TRAINING & EVALUATION" if args.train else "EVALUATION"
    pygame.display.set_caption(f"Agent {mode_text} — Vehicle {vehicle}")
    font_big = pygame.font.SysFont("consolas", 28, bold=True)
    font = pygame.font.SysFont("consolas", 18)

    lines = [
        f"AGENT {mode_text} — Vehicle {vehicle} ({'DQN' if vehicle == 'R' else 'SARSA'})",
        "",
        "The trained agent will drive autonomously.",
        f"Live Training: {'ENABLED (Exploring)' if args.train else 'DISABLED (Greedy)'}",
        "",
        "  ESC  — Quit at any time",
        "",
        "Press ENTER to start...",
    ]

    screen.fill((30, 30, 36))
    for i, line in enumerate(lines):
        f = font_big if i == 0 else font
        color = (0, 229, 255) if i == 0 else (200, 200, 200)
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

    # ── Load agent ───────────────────────────────────────────────
    checkpoint_dir = Path("artifacts/checkpoints")

    if vehicle == "R":
        agent = DQNAgent()
        try:
            agent.load(checkpoint_dir)
            print("✅ DQN checkpoint loaded.")
        except Exception as e:
            print(f"⚠️  No DQN checkpoint found ({e}). Agent will act randomly.")
    else:
        agent = SARSAAgent()
        try:
            agent.load(checkpoint_dir)
            print("✅ SARSA checkpoint loaded.")
        except Exception as e:
            print(f"⚠️  No SARSA checkpoint found ({e}). Agent will act randomly.")

    # Enable or disable eval mode based on --train flag
    agent.set_eval_mode(not args.train)

    # ── Run episode ───────────────────────────────────
    config_overrides = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
        "vehicles_density": args.vehicles_density
    }

    env_mgr = EnvManager(render_mode="human", config_overrides=config_overrides)
    result = env_mgr.reset()
    clock = pygame.time.Clock()
    
    update_count = 0
    total_loss = 0.0

    try:
        for step in range(args.duration):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

            # Agent picks action
            action = agent.act(
                state=result.raw_state,
                discrete_state=result.discrete_state,
            )
            
            # Step environment
            prev_result = result
            result = env_mgr.step(action)
            
            if step >= args.duration - 1 and not result.terminated:
                result.truncated = True

            # If training is enabled, record transition and update agent
            if args.train:
                transition = Transition(
                    step=step,
                    state=prev_result.raw_state.tolist(),
                    action=action,
                    reward=result.reward,
                    next_state=result.raw_state.tolist(),
                    terminated=result.terminated,
                    truncated=result.truncated,
                    discrete_state=prev_result.discrete_state,
                    next_discrete_state=result.discrete_state,
                    lane=result.lane_index,
                    speed=result.speed,
                )
                metrics = agent.update(transition)
                if metrics:
                    update_count += 1
                    if "loss" in metrics:
                        total_loss += metrics["loss"]
                    elif "td_error" in metrics:
                        total_loss += metrics["td_error"]

            clock.tick(60)  # Uncapped rendering or set to 60 for smoothness

            if result.terminated or result.truncated:
                break

        print(f"\n{'💥 Collision!' if result.terminated else '⏱️  Time limit reached.'}")
        print(f"Steps: {env_mgr.step_count}  |  Total reward: {env_mgr.total_reward:.2f}")
        
        if args.train and update_count > 0:
            print(f"Training Updates: {update_count}  |  Avg Loss/TD: {(total_loss/update_count):.4f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env_mgr.close()
        pygame.quit()
        
        # Save if requested
        if args.save:
            agent.save(checkpoint_dir)
            print("✅ Checkpoint saved successfully.")


if __name__ == "__main__":
    main()
