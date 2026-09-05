"""
Highway SVG Renderer.

Generates an HTML/CSS visualization of the highway environment
from the raw observation state. Designed to look like an actual
highway with proper lane markings, vehicle labels, and motion cues.
"""

from __future__ import annotations
import numpy as np


def render_highway_svg(
    raw_state: list[float],
    ego_lane: int,
    vehicle_type: str = "R",
    speed: float = 0.0,
    last_action: str = "",
    step_count: int = 0,
    reward: float = 0.0,
) -> str:
    """
    Generate HTML for the highway visualization.

    Parameters
    ----------
    raw_state : list[float]
        Flat 36-dim vector (6 vehicles × 6 features).
    ego_lane : int
        Ego vehicle's current lane index (0-3).
    vehicle_type : str
        'R' (DQN, cyan) or 'S' (SARSA, orange).
    speed : float
        Ego speed in m/s for HUD display.
    last_action : str
        Last action name for HUD display.
    step_count : int
        Current step count.
    reward : float
        Current step reward.
    """

    LANES = 4
    ROAD_HEIGHT = 400
    LANE_H = ROAD_HEIGHT // LANES  # 100px per lane
    EGO_X_PCT = 25  # Ego always at 25% from left

    ego_color = "#00E5FF" if vehicle_type == "R" else "#FF9100"
    ego_label = f"{vehicle_type} • {'DQN' if vehicle_type == 'R' else 'SARSA'}"

    # Build lane markings
    lane_lines_html = ""
    # Top edge (solid)
    lane_lines_html += f'<div style="position:absolute;top:0;left:0;width:100%;height:3px;background:#FFF;opacity:0.6;"></div>'
    # Bottom edge (solid)
    lane_lines_html += f'<div style="position:absolute;bottom:0;left:0;width:100%;height:3px;background:#FFF;opacity:0.6;"></div>'
    # Dashed center lines
    for i in range(1, LANES):
        y = i * LANE_H
        lane_lines_html += f'''<div style="
            position:absolute; top:{y}px; left:0; width:100%; height:2px;
            background:repeating-linear-gradient(to right, #FFF 0px, #FFF 30px, transparent 30px, transparent 60px);
            opacity:0.35;
        "></div>'''

    # Build vehicle HTML
    cars_html = ""

    if raw_state and len(raw_state) == 36:
        obs = np.array(raw_state).reshape((6, 6))

        for i, row in enumerate(obs):
            if np.allclose(row, 0.0) and i != 0:
                continue

            if i == 0:
                # Ego vehicle — fixed position
                left_pct = EGO_X_PCT
                lane = ego_lane
                color = ego_color
                border = f"3px solid {ego_color}"
                label_html = f'''<div style="
                    position:absolute; top:-22px; left:50%; transform:translateX(-50%);
                    font-size:11px; font-weight:bold; color:{ego_color};
                    white-space:nowrap; font-family:Consolas,monospace;
                    text-shadow: 0 0 6px rgba(0,0,0,0.8);
                ">{ego_label}</div>'''
            else:
                # NPC vehicle
                rel_x = row[0]
                rel_y = row[1]
                left_pct = EGO_X_PCT + (rel_x * 50)
                lane = ego_lane + (rel_y * 4)
                color = "#555"
                border = "1px solid #444"
                label_html = ""

            # Clamp
            if left_pct < -5 or left_pct > 105:
                continue

            top_px = (lane * LANE_H) + (LANE_H / 2)

            # Vehicle body: a rectangle with rounded corners + windshield
            car_w = 70
            car_h = 32

            cars_html += f'''<div style="
                position:absolute;
                left:{left_pct}%;
                top:{top_px}px;
                transform: translate(-50%, -50%);
                width:{car_w}px; height:{car_h}px;
                transition: top 0.15s ease-out;
                z-index: {'10' if i == 0 else '5'};
            ">
                {label_html}
                <div style="
                    width:100%; height:100%;
                    background: {color};
                    border-radius: 8px;
                    border: {border};
                    position: relative;
                    box-shadow: {'0 0 12px ' + ego_color + '40' if i == 0 else '0 1px 3px rgba(0,0,0,0.5)'};
                ">
                    <!-- Windshield -->
                    <div style="
                        position:absolute; top:4px; right:6px;
                        width:18px; height:{car_h - 8}px;
                        background: rgba(100,200,255,0.3);
                        border-radius: 4px;
                    "></div>
                    <!-- Headlights -->
                    <div style="
                        position:absolute; top:3px; left:-2px;
                        width:5px; height:5px; border-radius:50%;
                        background: {'#FFF' if i == 0 else '#AAA'};
                        opacity: 0.9;
                    "></div>
                    <div style="
                        position:absolute; bottom:3px; left:-2px;
                        width:5px; height:5px; border-radius:50%;
                        background: {'#FFF' if i == 0 else '#AAA'};
                        opacity: 0.9;
                    "></div>
                    <!-- Taillights -->
                    <div style="position:absolute;top:3px;right:-2px;width:5px;height:4px;border-radius:2px;background:#FF1744;opacity:0.8;"></div>
                    <div style="position:absolute;bottom:3px;right:-2px;width:5px;height:4px;border-radius:2px;background:#FF1744;opacity:0.8;"></div>
                </div>
            </div>'''

    # HUD overlay (inside the road, bottom-right)
    safety = "SAFE" if reward >= 0 else "DANGER"
    safety_color = "#4CAF50" if reward >= 0 else "#FF1744"
    action_display = last_action.upper() if last_action else "—"

    hud_html = f'''<div style="
        position:absolute; bottom:12px; right:12px;
        background: rgba(15,15,20,0.85);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: Consolas, monospace;
        font-size: 12px;
        color: #ccc;
        min-width: 180px;
        z-index: 20;
    ">
        <div style="color:{ego_color};font-weight:bold;margin-bottom:6px;">HUMAN → {vehicle_type}</div>
        <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="opacity:0.6;">Speed</span><span>{speed:.0f} m/s</span></div>
        <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="opacity:0.6;">Lane</span><span>{ego_lane}</span></div>
        <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="opacity:0.6;">Action</span><span style="color:{ego_color};">{action_display}</span></div>
        <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="opacity:0.6;">Reward</span><span>{reward:+.2f}</span></div>
        <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="opacity:0.6;">Steps</span><span>{step_count}</span></div>
        <div style="margin-top:4px;color:{safety_color};font-weight:bold;font-size:11px;">{safety}</div>
    </div>'''

    # Road scroll animation
    scroll_speed = max(0.3, 3.0 - (speed / 15.0)) if speed > 0 else 100

    return f'''
    <style>
    @keyframes roadScroll {{
        from {{ background-position: 0 0; }}
        to {{ background-position: -60px 0; }}
    }}
    </style>
    <div style="
        width:100%; height:{ROAD_HEIGHT}px;
        background-color: #2a2a30;
        background-image: repeating-linear-gradient(
            to right,
            transparent,
            transparent 58px,
            rgba(255,255,255,0.02) 58px,
            rgba(255,255,255,0.02) 60px
        );
        animation: roadScroll {scroll_speed}s linear infinite;
        position: relative;
        overflow: hidden;
        border-radius: 6px;
        border: 1px solid #444;
    ">
        {lane_lines_html}
        {cars_html}
        {hud_html}
    </div>'''
