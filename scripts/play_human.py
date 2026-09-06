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
from src.envs.actions import action_name


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
    result = mgr.start(vehicle=vehicle, config_overrides=config_overrides)
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("consolas", 16)
    hud_small = pygame.font.SysFont("consolas", 13)
    show_hud = True

    def draw_hud(current_result, status="DRIVING"):
        if not show_hud:
            return
        surface = pygame.display.get_surface()
        if surface is None:
            return
        width, height = surface.get_size()
        panel_height = 104
        panel = pygame.Surface((width, panel_height), pygame.SRCALPHA)
        panel.fill((22, 22, 28, 228))
        surface.blit(panel, (0, height - panel_height))
        surface.blit(
            hud_font.render(f"HUMAN {vehicle} LIVE DATA  |  {status}", True, (0, 229, 255)),
            (12, height - 96),
        )
        action = current_result.action_taken if current_result.action_taken is not None else 1
        line = (
            f"step={mgr.step_count:03d}  action={action_name(int(action)):<11} "
            f"reward={mgr.total_reward:>7.2f}  lane={current_result.lane_index}  "
            f"speed={current_result.speed:>5.1f} m/s ({current_result.speed * 3.6:>5.1f} km/h)"
        )
        surface.blit(hud_small.render(line, True, (235, 235, 235)), (12, height - 72))
        surface.blit(
            hud_small.render("ARROWS drive   H HUD   S save   X discard   ESC close", True, (210, 210, 215)),
            (12, height - 51),
        )

    try:
        while not mgr.is_done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    mgr.discard()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key == pygame.K_s:
                        path = mgr.save()
                        print(f"Saved demonstration from GUI control: {path}")
                        return
                    elif event.key == pygame.K_x:
                        mgr.discard()
                        print("Episode discarded from GUI control.")
                        return
            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                mgr.discard()
                print("ESC pressed. Episode discarded.")
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

            result = mgr.act(action_key)
            draw_hud(result)
            pygame.display.flip()
            clock.tick(10)

        print("\nEpisode finished. Use the GUI to save or discard it.")
        while mgr.is_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    mgr.discard()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key == pygame.K_s:
                        path = mgr.save()
                        print(f"Saved demonstration from GUI control: {path}")
                    elif event.key == pygame.K_x:
                        mgr.discard()
                        print("Episode discarded from GUI control.")
            draw_hud(result, "COMPLETE — S SAVE / X DISCARD")
            pygame.display.flip()
            clock.tick(30)
        print(f"✅ Successfully saved demonstration to: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted! Discarding episode...")
        mgr.discard()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
