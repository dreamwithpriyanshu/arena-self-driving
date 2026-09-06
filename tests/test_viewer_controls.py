"""Pure control tests for the native SARSA viewer presets."""

from scripts.play_agent import ROAD_MODES, cycle_road_mode, road_mode_name


def test_road_mode_cycle_updates_target_speed() -> None:
    settings = {"target_speed": ROAD_MODES[0][1]}
    assert cycle_road_mode(settings) is None
    assert settings["target_speed"] == ROAD_MODES[1][1]
    cycle_road_mode(settings)
    assert settings["target_speed"] == ROAD_MODES[2][1]
    cycle_road_mode(settings)
    assert settings["target_speed"] == ROAD_MODES[0][1]


def test_road_mode_name_uses_nearest_preset() -> None:
    assert road_mode_name(12.2) == "CITY"
    assert road_mode_name(18.4) == "HIGHWAY"
    assert road_mode_name(23.8) == "EXPRESS"
