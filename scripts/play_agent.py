"""Responsive native PyGame viewer for one trained autonomous agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from src.agents.sarsa import SARSAAgent
from src.data.schemas import Transition
from src.envs.actions import action_name
from src.simulation.env_manager import EnvManager
from src.storage import checkpoints_dir, ensure_storage_dirs


RENDER_FPS = 60
POLICY_HZ = 15
MIN_TARGET_SPEED = 8.0
MAX_TARGET_SPEED = 40.0
ROAD_MODES = (("CITY", 12.0), ("HIGHWAY", 18.0), ("EXPRESS", 24.0))


def road_mode_name(speed: float) -> str:
    return min(ROAD_MODES, key=lambda mode: abs(mode[1] - speed))[0]


def cycle_road_mode(settings: dict) -> None:
    current = min(range(len(ROAD_MODES)), key=lambda index: abs(ROAD_MODES[index][1] - settings["target_speed"]))
    settings["target_speed"] = ROAD_MODES[(current + 1) % len(ROAD_MODES)][1]


def show_setup(settings: dict, training: bool) -> bool:
    screen = pygame.display.set_mode((740, 350))
    pygame.display.set_caption("Agent Viewer - SARSA")
    title = pygame.font.SysFont("consolas", 26, bold=True)
    font = pygame.font.SysFont("consolas", 17)
    clock = pygame.time.Clock()
    while True:
        screen.fill((24, 26, 33))
        lines = [
            ("SARSA AUTONOMOUS VIEWER", (0, 229, 255), title),
            (f"Mode: {'LIVE TRAINING' if training else 'GREEDY EVALUATION'}", (255, 166, 77), font),
            (f"Traffic: {settings['vehicles_count']} NPCs  |  density: {settings['vehicles_density']:.1f}", (235, 235, 235), font),
            (f"Road pace: {road_mode_name(settings['target_speed'])}  |  target speed: {settings['target_speed']:.1f} m/s", (235, 235, 235), font),
            ("Setup: UP/DOWN traffic  LEFT/RIGHT density  +/- speed  [/] duration  / road mode", (175, 182, 194), font),
            ("Drive: SPACE/P pause  +/- speed  [/] duration  / road mode  H HUD  ESC quit"
             + ("  Q/E epsilon" if training else ""), (175, 182, 194), font),
            ("ENTER start", (0, 229, 255), font),
        ]
        for index, (line, color, line_font) in enumerate(lines):
            screen.blit(line_font.render(line, True, color), (26, 24 + index * 42))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return True
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_UP:
                    settings["vehicles_count"] = min(60, settings["vehicles_count"] + 1)
                elif event.key == pygame.K_DOWN:
                    settings["vehicles_count"] = max(4, settings["vehicles_count"] - 1)
                elif event.key == pygame.K_RIGHT:
                    settings["vehicles_density"] = min(3.0, round(settings["vehicles_density"] + 0.1, 1))
                elif event.key == pygame.K_LEFT:
                    settings["vehicles_density"] = max(0.2, round(settings["vehicles_density"] - 0.1, 1))
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    settings["target_speed"] = min(MAX_TARGET_SPEED, settings["target_speed"] + 1.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    settings["target_speed"] = max(MIN_TARGET_SPEED, settings["target_speed"] - 1.0)
                elif event.key == pygame.K_LEFTBRACKET:
                    settings["duration"] = max(10, settings["duration"] - 10)
                elif event.key == pygame.K_RIGHTBRACKET:
                    settings["duration"] = min(3600, settings["duration"] + 10)
                elif event.key == pygame.K_SLASH:
                    cycle_road_mode(settings)
        clock.tick(RENDER_FPS)


def tune_vehicle(env_mgr: EnvManager, target_speed: float) -> None:
    env = env_mgr._env.unwrapped
    for driver in getattr(env, "controlled_vehicles", []):
        driver.color = (255, 166, 77)
        driver.LENGTH, driver.WIDTH = 6.5, 2.6
        driver.speed, driver.target_speed = target_speed, target_speed


def balance_viewer_traffic(env_mgr: EnvManager) -> None:
    """Put part of the NPC traffic behind the ego for a two-sided scene."""
    env = env_mgr._env.unwrapped
    ego = getattr(env, "vehicle", None)
    road = getattr(env, "road", None)
    vehicles = list(getattr(road, "vehicles", [])) if road is not None else []
    if ego is None or road is None or len(vehicles) < 3:
        return

    ego_longitudinal = float(ego.position[0])
    npcs = [vehicle for vehicle in vehicles if vehicle is not ego]
    for index, vehicle in enumerate(npcs[len(npcs) // 2:]):
        lane = road.network.get_lane(vehicle.lane_index)
        longitudinal = ego_longitudinal - 24.0 - index * 22.0
        vehicle.position = lane.position(longitudinal, 0.0)
        vehicle.heading = lane.heading_at(longitudinal)
        vehicle.speed = max(8.0, min(float(vehicle.speed), MAX_TARGET_SPEED))


def main() -> None:
    parser = argparse.ArgumentParser(description="Responsive native viewer for the SARSA policy")
    parser.add_argument("--vehicles-count", type=int, default=15)
    parser.add_argument("--duration", type=int, default=120, help="Episode duration in real-time seconds")
    parser.add_argument("--vehicles-density", type=float, default=1.0)
    parser.add_argument("--target-speed", type=float, default=18.0)
    parser.add_argument("--train", action="store_true", help="Train while viewing")
    parser.add_argument("--save", action="store_true", help="Save checkpoint on exit")
    args = parser.parse_args()
    settings = {"vehicles_count": args.vehicles_count, "vehicles_density": args.vehicles_density,
                "duration": args.duration, "target_speed": args.target_speed}

    pygame.init()
    if not show_setup(settings, args.train):
        pygame.quit()
        return
    pygame.display.quit()

    agent = SARSAAgent()
    ensure_storage_dirs()
    checkpoint_dir = checkpoints_dir()
    try:
        agent.load(checkpoint_dir)
        print("SARSA checkpoint loaded.")
    except Exception as error:
        print(f"No SARSA checkpoint found ({error}). Agent will act randomly.")
    agent.set_eval_mode(not args.train)

    env_mgr = EnvManager(render_mode="rgb_array", config_overrides={
        "vehicles_count": settings["vehicles_count"], "vehicles_density": settings["vehicles_density"],
        "duration": settings["duration"], "simulation_frequency": 30, "policy_frequency": POLICY_HZ,
        "screen_height": 390,
    })
    result = env_mgr.reset()
    tune_vehicle(env_mgr, settings["target_speed"])
    balance_viewer_traffic(env_mgr)
    pygame.display.set_mode((600, 390))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    paused, show_hud, elapsed = False, True, 0.0
    action, updates, loss_total = 1, 0, 0.0
    button_bounds: dict[str, pygame.Rect] = {}

    def set_target_speed(delta: float = 0.0, cycle: bool = False) -> None:
        if cycle:
            cycle_road_mode(settings)
        else:
            settings["target_speed"] = min(MAX_TARGET_SPEED, max(MIN_TARGET_SPEED, settings["target_speed"] + delta))
        for driver in getattr(env_mgr._env.unwrapped, "controlled_vehicles", []):
            driver.target_speed = settings["target_speed"]
        if cycle:
            for vehicle in getattr(env_mgr._env.unwrapped.road, "vehicles", []):
                vehicle.target_speed = settings["target_speed"]

    def set_duration(delta: int) -> None:
        settings["duration"] = min(3600, max(10, settings["duration"] + delta))
        env_mgr._env.unwrapped.config["duration"] = settings["duration"]

    def set_epsilon(delta: float) -> None:
        if args.train:
            agent.epsilon = min(1.0, max(agent.epsilon_end, agent.epsilon + delta))

    def draw_hud(status: str) -> None:
        button_bounds.clear()
        if not show_hud:
            return
        surface = pygame.display.get_surface()
        if surface is None:
            return
        width, height = surface.get_size()
        panel = pygame.Surface((width, 120), pygame.SRCALPHA)
        panel.fill((18, 20, 27, 232))
        surface.blit(panel, (0, height - 120))
        accent = (255, 196, 77) if paused else (0, 229, 255)
        surface.blit(font.render(f"SARSA AUTONOMOUS | {status}", True, accent), (12, height - 112))
        surface.blit(font.render(
            f"ROAD {road_mode_name(settings['target_speed'])} | TARGET SPEED {settings['target_speed']:.1f} m/s",
            True,
            (255, 196, 77),
        ), (12, height - 88))
        metrics = f"action={action_name(int(action)):<10} step={env_mgr.step_count:04d} reward={env_mgr.total_reward:7.2f} lane={result.lane_index} speed={result.speed:5.1f} duration={settings['duration']}s eps={agent.epsilon:.2f}"
        surface.blit(font.render(metrics, True, (238, 238, 240)), (12, height - 64))
        controls = "click controls or SPACE/P +/- TIME / | H HUD | ESC quit"
        if args.train:
            controls = "click controls or SPACE/P +/- TIME Q/E epsilon / | H HUD | ESC quit"
        surface.blit(font.render(controls, True, (185, 191, 204)), (12, height - 38))
        x = width - 12
        buttons = [
            ("pause", "RESUME" if paused else "PAUSE"),
            ("slower", "- SPD"),
            ("faster", "+ SPD"),
            ("shorter", "-TIME"),
            ("longer", "+TIME"),
        ]
        if args.train:
            buttons.extend([("epsilon_down", "- EPS"), ("epsilon_up", "+ EPS")])
        buttons.append(("mode", road_mode_name(settings["target_speed"])))
        for button_id, label in reversed(buttons):
            button = pygame.Rect(x - 60, height - 114, 56, 22)
            pygame.draw.rect(surface, (53, 62, 76), button, border_radius=3)
            pygame.draw.rect(surface, accent, button, width=1, border_radius=3)
            surface.blit(font.render(label, True, (238, 238, 240)), font.render(label, True, (238, 238, 240)).get_rect(center=button.center))
            button_bounds[button_id] = button
            x -= 62

    cached_frame = None

    def render_scene(force: bool = False) -> None:
        nonlocal cached_frame
        if cached_frame is None or force:
            cached_frame = env_mgr._env.render()
        surface = pygame.display.get_surface()
        if cached_frame is not None and surface is not None:
            image = pygame.surfarray.make_surface(cached_frame.swapaxes(0, 1))
            surface.blit(image, (0, 0))

    try:
        while not (result.terminated or result.truncated):
            frame_seconds = clock.tick(RENDER_FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    if event.key in (pygame.K_p, pygame.K_SPACE):
                        paused, elapsed = not paused, 0.0
                    elif event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        set_target_speed(1.0)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        set_target_speed(-1.0)
                    elif event.key == pygame.K_LEFTBRACKET:
                        set_duration(-10)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        set_duration(10)
                    elif event.key == pygame.K_SLASH:
                        set_target_speed(cycle=True)
                    elif event.key == pygame.K_q:
                        set_epsilon(-0.05)
                    elif event.key == pygame.K_e:
                        set_epsilon(0.05)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if button_bounds.get("pause", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        paused, elapsed = not paused, 0.0
                    elif button_bounds.get("slower", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_target_speed(-1.0)
                    elif button_bounds.get("faster", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_target_speed(1.0)
                    elif button_bounds.get("shorter", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_duration(-10)
                    elif button_bounds.get("longer", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_duration(10)
                    elif button_bounds.get("mode", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_target_speed(cycle=True)
                    elif button_bounds.get("epsilon_down", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_epsilon(-0.05)
                    elif button_bounds.get("epsilon_up", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        set_epsilon(0.05)
            if not paused:
                elapsed += frame_seconds
                while elapsed >= 1 / POLICY_HZ and not (result.terminated or result.truncated):
                    action = agent.act(result.raw_state, result.discrete_state)
                    previous = result
                    result = env_mgr.step(action)
                    cached_frame = None
                    if args.train:
                        metrics = agent.update(Transition(step=env_mgr.step_count - 1, state=previous.raw_state.tolist(), action=action, reward=result.reward, next_state=result.raw_state.tolist(), terminated=result.terminated, truncated=result.truncated, discrete_state=previous.discrete_state, next_discrete_state=result.discrete_state, lane=result.lane_index, speed=result.speed))
                        if metrics:
                            updates += 1
                            loss_total += float(metrics.get("loss", metrics.get("td_error", 0.0)))
                    elapsed -= 1 / POLICY_HZ
            render_scene()
            draw_hud("PAUSED" if paused else "DRIVING")
            pygame.display.flip()
        print(f"Episode ended after {env_mgr.step_count} steps; reward={env_mgr.total_reward:.2f}.")
        if args.train and updates:
            print(f"Training updates={updates}; average loss/TD={loss_total / updates:.4f}")
        finished = True
        while finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    finished = False
            draw_hud("COMPLETE - ESC closes")
            surface = pygame.display.get_surface()
            if surface is not None:
                surface.blit(font.render("EPISODE COMPLETE - ESC closes this viewer", True, (255, 196, 77)), (12, 12))
            pygame.display.flip()
            clock.tick(RENDER_FPS)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env_mgr.close()
        pygame.quit()
        if args.save:
            agent.save(checkpoint_dir)
            print("Checkpoint saved.")


if __name__ == "__main__":
    main()
