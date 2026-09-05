"""
Standalone PyGame script for real-time Agent Evaluation.
Watch the trained DQN or SARSA agent drive autonomously at 60 FPS!

Bypasses the orchestrator/facade to avoid circular imports.
Imports agents and env_manager directly.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.simulation.env_manager import EnvManager


def main():
    parser = argparse.ArgumentParser(description="Agent Evaluation Interface")
    parser.add_argument("vehicle", type=str, choices=["R", "S"], help="Vehicle agent to watch: R (DQN) or S (SARSA)")
    parser.add_argument("--vehicles-count", type=int, default=15, help="Number of NPC vehicles on the road")
    parser.add_argument("--duration", type=int, default=120, help="Max duration of the episode in steps")
    parser.add_argument("--vehicles-density", type=float, default=1.0, help="Traffic density multiplier")
    
    args = parser.parse_args()
    vehicle = args.vehicle.upper()

    pygame.init()

    # ── Pre-game instruction screen ──────────────────────────────
    screen = pygame.display.set_mode((640, 300))
    pygame.display.set_caption(f"Agent Evaluation — Vehicle {vehicle}")
    font_big = pygame.font.SysFont("consolas", 28, bold=True)
    font = pygame.font.SysFont("consolas", 18)

    lines = [
        f"AGENT EVALUATION — Vehicle {vehicle} ({'DQN' if vehicle == 'R' else 'SARSA'})",
        "",
        "The trained agent will drive autonomously.",
        "A PyGame window will open with the highway simulation.",
        "",
        "  ESC  — Quit at any time",
        "",
        "Press ENTER to start...",
    ]

    screen.fill((30, 30, 36))
    for i, line in enumerate(lines):
        f = font_big if i == 0 else font
        color = (0, 229, 255) if i == 0 else (200, 200, 200)
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

    # Close the instruction window — highway-env will create its own
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

    agent.set_eval_mode(True)

    # ── Run evaluation episode ───────────────────────────────────
    config_overrides = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
        "vehicles_density": args.vehicles_density
    }

    env_mgr = EnvManager(render_mode="human", config_overrides=config_overrides)
    result = env_mgr.reset()
    clock = pygame.time.Clock()

    try:
        for step in range(args.duration):
            # Process events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

            action = agent.act(
                state=result.raw_state,
                discrete_state=result.discrete_state,
            )
            result = env_mgr.step(action)

            clock.tick(5)  # 5 decisions per second — readable speed

            if result.terminated or result.truncated:
                break

        print(f"\n{'💥 Collision!' if result.terminated else '⏱️  Time limit reached.'}")
        print(f"Steps: {env_mgr.step_count}  |  Total reward: {env_mgr.total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env_mgr.close()
        pygame.quit()


if __name__ == "__main__":
    main()
