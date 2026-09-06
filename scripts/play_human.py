"""Native PyGame human-driving recorder with responsive held-key controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from src.envs.actions import Action, action_name
from src.human.control_bindings import active_actions, load_control_profiles
from src.human.episode_manager import EpisodeManager


RENDER_FPS = 60
DECISION_HZ = 15
ROAD_MODES = (("CITY", 12.0), ("HIGHWAY", 18.0), ("EXPRESS", 24.0))
PYGAME_KEYS = {
    "left": pygame.K_LEFT, "right": pygame.K_RIGHT,
    "up": pygame.K_UP, "down": pygame.K_DOWN,
    "space": pygame.K_SPACE,
}


def profile_summary(profile: dict[Action, tuple[str, ...]]) -> str:
    def keys(action: Action) -> str:
        return "/".join(key.upper() for key in profile[action]) or "-"
    return f"{keys(Action.FASTER)} accelerate   {keys(Action.SLOWER)} brake   {keys(Action.LANE_LEFT)}/{keys(Action.LANE_RIGHT)} lanes"


def cycle_road_mode(settings: dict) -> str:
    """Select the next realistic target-speed preset."""
    current = min(range(len(ROAD_MODES)), key=lambda index: abs(ROAD_MODES[index][1] - settings["target_speed"]))
    name, speed = ROAD_MODES[(current + 1) % len(ROAD_MODES)]
    settings["target_speed"] = speed
    return name


def road_mode_name(speed: float) -> str:
    return min(ROAD_MODES, key=lambda mode: abs(mode[1] - speed))[0]


def show_setup(vehicle: str, settings: dict, profiles: dict[str, dict[Action, tuple[str, ...]]], profile_name: str) -> tuple[bool, str]:
    """Show editable pre-drive settings; return start choice and profile."""
    screen = pygame.display.set_mode((740, 380))
    pygame.display.set_caption(f"Human Training - Vehicle {vehicle}")
    title = pygame.font.SysFont("consolas", 27, bold=True)
    font = pygame.font.SysFont("consolas", 17)
    clock = pygame.time.Clock()
    while True:
        screen.fill((24, 26, 33))
        lines = [
            (f"HUMAN TRAINING - VEHICLE {vehicle}", (0, 229, 255), title),
            ("Configure the road, then press ENTER to drive.", (220, 223, 230), font),
            (f"Traffic: {settings['vehicles_count']} NPCs  |  density: {settings['vehicles_density']:.1f}", (235, 235, 235), font),
            (f"Initial speed: {settings['target_speed']:.1f} m/s  |  duration: {settings['duration']} s", (235, 235, 235), font),
            (f"Controls: {profile_name.upper()}  -  {profile_summary(profiles[profile_name])}", (255, 166, 77), font),
            ("UP/DOWN NPCs  LEFT/RIGHT density  +/- speed  / road mode", (175, 182, 194), font),
            ("In drive: P pause  H HUD  +/- speed  / road mode  ESC discard", (175, 182, 194), font),
            ("ENTER start    ESC quit", (0, 229, 255), font),
        ]
        for index, (line, color, line_font) in enumerate(lines):
            screen.blit(line_font.render(line, True, color), (26, 26 + index * 39))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, profile_name
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return True, profile_name
                if event.key == pygame.K_ESCAPE:
                    return False, profile_name
                if event.key == pygame.K_UP:
                    settings["vehicles_count"] = min(60, settings["vehicles_count"] + 1)
                elif event.key == pygame.K_DOWN:
                    settings["vehicles_count"] = max(4, settings["vehicles_count"] - 1)
                elif event.key == pygame.K_RIGHT:
                    settings["vehicles_density"] = min(3.0, round(settings["vehicles_density"] + 0.1, 1))
                elif event.key == pygame.K_LEFT:
                    settings["vehicles_density"] = max(0.2, round(settings["vehicles_density"] - 0.1, 1))
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    settings["target_speed"] = min(30.0, settings["target_speed"] + 1.0)
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                    settings["target_speed"] = max(8.0, settings["target_speed"] - 1.0)
                elif event.key == pygame.K_SLASH:
                    cycle_road_mode(settings)
        clock.tick(RENDER_FPS)


def make_driver_visible(mgr: EpisodeManager, vehicle: str, target_speed: float) -> None:
    """Give the human vehicle an unmistakable colour and a modest size boost."""
    env = mgr._env_mgr._env.unwrapped  # Native renderer needs the controlled vehicle.
    for driver in getattr(env, "controlled_vehicles", []):
        driver.color = (0, 229, 255)
        driver.LENGTH = 6.5
        driver.WIDTH = 2.6
        driver.speed = target_speed
        driver.target_speed = target_speed


def main() -> None:
    parser = argparse.ArgumentParser(description="Responsive native PyGame human-training recorder")
    parser.add_argument("--vehicles-count", type=int, default=15, help="NPC vehicles (default: 15)")
    parser.add_argument("--duration", type=int, default=120, help="Episode duration in real-time seconds (default: 120)")
    parser.add_argument("--vehicles-density", type=float, default=1.0, help="Traffic density multiplier (default: 1.0)")
    parser.add_argument("--target-speed", type=float, default=18.0, help="Initial speed in m/s (default: 18.0)")
    args = parser.parse_args()
    # One-policy application: demonstrations are always for SARSA.
    args.vehicle = "S"

    settings = {"vehicles_count": args.vehicles_count, "vehicles_density": args.vehicles_density,
                "duration": args.duration, "target_speed": args.target_speed}
    default_profile, profiles = load_control_profiles()

    pygame.init()
    started, profile_name = show_setup(args.vehicle, settings, profiles, default_profile)
    if not started:
        pygame.quit()
        return
    pygame.display.quit()

    config = {
        "vehicles_count": settings["vehicles_count"], "vehicles_density": settings["vehicles_density"],
        "duration": settings["duration"], "initial_speed": settings["target_speed"],
        "simulation_frequency": 30, "policy_frequency": DECISION_HZ, "screen_height": 390,
    }
    mgr = EpisodeManager(render_mode="human", max_steps=settings["duration"] * DECISION_HZ)
    result = mgr.start(vehicle=args.vehicle, config_overrides=config)
    make_driver_visible(mgr, args.vehicle, settings["target_speed"])

    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("consolas", 15)
    show_hud, paused = True, False
    decision_elapsed, action_turn = 0.0, 0
    last_applied = Action.IDLE
    button_bounds: dict[str, pygame.Rect] = {}

    def adjust_target_speed(delta: float) -> None:
        settings["target_speed"] = min(30.0, max(8.0, settings["target_speed"] + delta))
        env = mgr._env_mgr._env.unwrapped
        for driver in getattr(env, "controlled_vehicles", []):
            driver.target_speed = settings["target_speed"]

    def draw_hud(held: list[Action]) -> None:
        button_bounds.clear()
        if not show_hud:
            return
        surface = pygame.display.get_surface()
        if surface is None:
            return
        width, height = surface.get_size()
        panel_height = 94
        panel = pygame.Surface((width, panel_height), pygame.SRCALPHA)
        panel.fill((18, 20, 27, 232))
        surface.blit(panel, (0, height - panel_height))
        status = "PAUSED" if paused else "DRIVING"
        held_label = "+".join(action_name(action) for action in held) if held else "IDLE"
        accent = (255, 196, 77) if paused else (0, 229, 255)
        surface.blit(hud_font.render(f"HUMAN {args.vehicle} | {status} | INPUT: {held_label}", True, accent), (12, height - 84))
        stats = (f"applied={action_name(last_applied):<10}  step={mgr.step_count:04d}  "
                 f"reward={mgr.total_reward:7.2f}  lane={result.lane_index}  "
                 f"speed={result.speed:5.1f} m/s  target={settings['target_speed']:4.1f}")
        surface.blit(hud_font.render(stats, True, (238, 238, 240)), (12, height - 60))
        controls = f"{profile_name.upper()} drive | click controls or P +/- / | H HUD | ESC discard"
        surface.blit(hud_font.render(controls, True, (185, 191, 204)), (12, height - 34))
        labels = [("pause", "RESUME" if paused else "PAUSE"), ("slower", "- SPD"), ("faster", "+ SPD"), ("mode", road_mode_name(settings["target_speed"]))]
        x = width - 12
        for button_id, label in reversed(labels):
            button = pygame.Rect(x - 76, height - 88, 70, 22)
            pygame.draw.rect(surface, (53, 62, 76), button, border_radius=3)
            pygame.draw.rect(surface, accent, button, width=1, border_radius=3)
            text = hud_font.render(label, True, (238, 238, 240))
            surface.blit(text, text.get_rect(center=button.center))
            button_bounds[button_id] = button
            x -= 82

    try:
        while not mgr.is_done:
            elapsed = clock.tick(RENDER_FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    mgr.discard()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        mgr.discard()
                        return
                    if event.key == pygame.K_p:
                        paused = not paused
                        decision_elapsed = 0.0
                    elif event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                        adjust_target_speed(-1.0)
                    elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        adjust_target_speed(1.0)
                    elif event.key == pygame.K_SLASH:
                        cycle_road_mode(settings)
                        adjust_target_speed(0.0)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if button_bounds.get("pause", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        paused = not paused
                        decision_elapsed = 0.0
                    elif button_bounds.get("slower", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        adjust_target_speed(-1.0)
                    elif button_bounds.get("faster", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        adjust_target_speed(1.0)
                    elif button_bounds.get("mode", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        cycle_road_mode(settings)
                        adjust_target_speed(0.0)

            keys = pygame.key.get_pressed()
            pressed = {name for name, keycode in PYGAME_KEYS.items() if keys[keycode]}
            held = active_actions(profiles[profile_name], pressed)
            if not paused:
                decision_elapsed += elapsed
                while decision_elapsed >= 1 / DECISION_HZ and not mgr.is_done:
                    action = held[action_turn % len(held)] if held else Action.IDLE
                    action_turn += 1
                    result = mgr.act_action(action)
                    last_applied = action
                    decision_elapsed -= 1 / DECISION_HZ
            draw_hud(held)
            pygame.display.flip()

        while mgr.is_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    mgr.discard()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    path = mgr.save()
                    print(f"Saved demonstration: {path}")
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    mgr.discard()
                    print("Episode discarded.")
                    return
            draw_hud([])
            surface = pygame.display.get_surface()
            if surface is not None:
                surface.blit(hud_font.render("COMPLETE: S save demonstration | X discard | ESC close", True, (255, 196, 77)), (12, 12))
            pygame.display.flip()
            clock.tick(RENDER_FPS)
    except KeyboardInterrupt:
        if mgr.is_active:
            mgr.discard()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
