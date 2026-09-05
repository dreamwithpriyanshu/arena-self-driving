"""
Standalone PyGame script for real-time Human Training.
Run this instead of Streamlit for a smooth driving experience!

Controls are shown on a pre-game instruction screen before the
highway window opens.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from src.human.episode_manager import EpisodeManager


def show_instructions(vehicle: str) -> bool:
    """
    Show a PyGame instruction screen.  Returns True if the user
    presses ENTER to start, False if they close / press ESC.
    """
    screen = pygame.display.set_mode((660, 340))
    pygame.display.set_caption(f"Human Training — Vehicle {vehicle}")
    font_big = pygame.font.SysFont("consolas", 28, bold=True)
    font = pygame.font.SysFont("consolas", 18)

    lines = [
        f"HUMAN TRAINING — Vehicle {vehicle} ({'DQN' if vehicle == 'R' else 'SARSA'})",
        "",
        "Drive the car and record a demonstration",
        "for the AI to learn from.",
        "",
        "  ↑  Arrow Up     — Accelerate",
        "  ↓  Arrow Down   — Brake",
        "  ←  Arrow Left   — Change lane left",
        "  →  Arrow Right  — Change lane right",
        "  ESC              — Quit & discard",
        "",
        "Press ENTER to start driving...",
    ]

    screen.fill((30, 30, 36))
    for i, line in enumerate(lines):
        f = font_big if i == 0 else font
        color = (255, 145, 0) if i == 0 else (200, 200, 200)
        if "ENTER" in line:
            color = (0, 229, 255)
        surf = f.render(line, True, color)
        screen.blit(surf, (30, 20 + i * 26))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return True
                if event.key == pygame.K_ESCAPE:
                    return False


def main():
    parser = argparse.ArgumentParser(description="Human Training Interface")
    parser.add_argument("vehicle", type=str, choices=["R", "S"], help="Vehicle to record data for: R (DQN) or S (SARSA)")
    parser.add_argument("--vehicles-count", type=int, default=15, help="Number of NPC vehicles on the road")
    parser.add_argument("--duration", type=int, default=120, help="Max duration of the episode in steps")
    parser.add_argument("--vehicles-density", type=float, default=1.0, help="Traffic density multiplier")
    
    args = parser.parse_args()
    vehicle = args.vehicle.upper()

    pygame.init()

    # ── Show instructions and wait for ENTER ─────────────────────
    if not show_instructions(vehicle):
        pygame.quit()
        return

    # Close the instruction window — highway-env creates its own
    pygame.display.quit()

    # ── Run the episode ──────────────────────────────────────────
    config_overrides = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
        "vehicles_density": args.vehicles_density
    }

    mgr = EpisodeManager(render_mode="human", max_steps=args.duration)
    mgr.start(vehicle=vehicle, config_overrides=config_overrides)
    clock = pygame.time.Clock()

    try:
        while not mgr.is_done:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                print("ESC pressed. Discarding episode...")
                mgr.discard()
                pygame.quit()
                return

            action_key = " "
            if keys[pygame.K_LEFT]:
                action_key = "arrowleft"
            elif keys[pygame.K_RIGHT]:
                action_key = "arrowright"
            elif keys[pygame.K_UP]:
                action_key = "arrowup"
            elif keys[pygame.K_DOWN]:
                action_key = "arrowdown"

            mgr.act(action_key)
            clock.tick(5)  # Cap at 5 decisions/sec — smooth but readable

        print("\nEpisode finished!")
        path = mgr.save()
        print(f"✅ Successfully saved demonstration to: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted! Discarding episode...")
        mgr.discard()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
