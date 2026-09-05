"""
Standalone PyGame script for real-time Agent Evaluation.
Watch the trained DQN or SARSA agent drive autonomously at 60 FPS!

Bypasses the orchestrator/facade to avoid circular imports.
Imports agents and env_manager directly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.simulation.env_manager import EnvManager


def main():
    vehicle = sys.argv[1].upper() if len(sys.argv) > 1 else "R"

    if vehicle not in ("R", "S"):
        print("Usage: python scripts/play_agent.py [R|S]")
        print("  R = Watch the DQN agent drive")
        print("  S = Watch the SARSA agent drive")
        sys.exit(1)

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
    env_mgr = EnvManager(render_mode="human")
    result = env_mgr.reset()
    clock = pygame.time.Clock()

    try:
        for step in range(1000):
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
