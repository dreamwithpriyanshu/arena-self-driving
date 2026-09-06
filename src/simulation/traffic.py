"""Traffic maintenance for the native viewer scenes."""

from __future__ import annotations

from typing import Any


VIEW_DISTANCE = 150.0
RESPAWN_AHEAD = 92.0
RESPAWN_BEHIND = 78.0
MIN_NPC_SPEED = 8.0
MAX_NPC_SPEED = 32.0


def maintain_viewer_traffic(env_manager: Any, target_speed: float) -> None:
    """Keep a two-sided, steady traffic stream around the ego vehicle.

    HighwayEnv vehicles remain physically simulated, but a native camera only
    shows a finite local area. Vehicles that leave that area are recycled into
    their existing lane instead of allowing the scene to become empty.
    """
    env = getattr(getattr(env_manager, "_env", None), "unwrapped", None)
    road = getattr(env, "road", None)
    ego = getattr(env, "vehicle", None)
    if road is None or ego is None:
        return

    vehicles = [vehicle for vehicle in getattr(road, "vehicles", []) if vehicle is not ego]
    if not vehicles:
        return

    ego_longitudinal = float(ego.position[0])
    for index, vehicle in enumerate(vehicles):
        vehicle.enable_lane_change = False
        vehicle.target_speed = max(MIN_NPC_SPEED, min(float(target_speed) * (0.9 + (index % 3) * 0.05), MAX_NPC_SPEED))

        relative_position = float(vehicle.position[0]) - ego_longitudinal
        if abs(relative_position) <= VIEW_DISTANCE and not getattr(vehicle, "crashed", False):
            continue

        lane = road.network.get_lane(vehicle.lane_index)
        behind = index % 2 == 1
        longitudinal = ego_longitudinal - RESPAWN_BEHIND - index * 8.0 if behind else ego_longitudinal + RESPAWN_AHEAD + index * 8.0
        vehicle.position = lane.position(longitudinal, 0.0)
        vehicle.heading = lane.heading_at(longitudinal)
        vehicle.speed = vehicle.target_speed
        vehicle.crashed = False
        vehicle.lane_index = vehicle.lane_index
        vehicle.target_lane_index = vehicle.lane_index
