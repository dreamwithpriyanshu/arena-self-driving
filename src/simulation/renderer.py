"""
Custom SVG Renderer for the Live Tracking view.

Takes the raw environment state (36-dim vector) and ego lane,
and generates an SVG string representing the highway from a top-down perspective.
Replaces the heavy PyGame video renderer for a fast, responsive Streamlit view.
"""

from __future__ import annotations

import numpy as np

# A basic CSS + SVG template for the road
SVG_TEMPLATE = """
<style>
@keyframes scrollRoad {{
    from {{ background-position: 0 0; }}
    to {{ background-position: -40px 0; }}
}}
.road-line {{
    position: absolute; width: 100%; height: 2px;
    background-image: linear-gradient(to right, #FFF 50%, transparent 50%);
    background-size: 40px 100%; opacity: 0.3;
    animation: scrollRoad 0.5s linear infinite;
}}
</style>
<div style="width: 100%; height: 300px; background-color: #1E1E24; position: relative; overflow: hidden; border-radius: 8px; border: 1px solid #333;">
    <!-- Road lines -->
    <div class="road-line" style="top: 25%;"></div>
    <div class="road-line" style="top: 50%;"></div>
    <div class="road-line" style="top: 75%;"></div>
    
    {cars_html}
</div>
"""

CAR_TEMPLATE = """
<div style="position: absolute; left: {left}%; top: {top}%; width: 60px; height: 30px; transform: translate(-50%, -50%); transition: left 0.3s ease-in-out, top 0.3s ease-in-out;">
    {svg_content}
</div>
"""


def get_car_svg(color: str) -> str:
    """Returns an inline SVG for a car facing right."""
    # Rotate the vertical top-down SVG 90 degrees clockwise so it drives to the right
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 200" style="transform: rotate(90deg); width: 100%; height: 100%;">
        <rect x="25" y="15" width="50" height="170" rx="15" fill="#000000" opacity="0.3" />
        <rect x="20" y="10" width="60" height="180" rx="15" fill="{color}" />
        <rect x="25" y="50" width="50" height="100" rx="10" fill="rgba(0,0,0,0.2)" />
        <path d="M 30 55 L 70 55 L 75 75 L 25 75 Z" fill="#1A1A24" />
        <path d="M 30 145 L 70 145 L 75 125 L 25 125 Z" fill="#1A1A24" />
        <rect x="25" y="12" width="12" height="6" rx="3" fill="#FFFFFF" opacity="0.9"/>
        <rect x="63" y="12" width="12" height="6" rx="3" fill="#FFFFFF" opacity="0.9"/>
        <rect x="25" y="182" width="15" height="5" rx="2" fill="#FF1744" />
        <rect x="60" y="182" width="15" height="5" rx="2" fill="#FF1744" />
    </svg>
    """


def render_highway_svg(raw_state: list[float], ego_lane: int, vehicle_type: str = "R") -> str:
    """
    Given a flat 36-dim raw_state vector (6 vehicles * 6 features),
    generate the HTML/SVG representation.
    Features: [x, y, vx, vy, cos_h, sin_h]
    """
    if not raw_state or len(raw_state) != 36:
        return SVG_TEMPLATE.format(cars_html="")

    obs = np.array(raw_state).reshape((6, 6))
    
    # Constants for projection
    LANE_HEIGHT_PERCENT = 25  # 4 lanes -> 25% height per lane
    # Map normalized X to percentage width. Ego is at ~30% from the left to see ahead.
    # The view window is roughly [-1, 2] in normalized coordinates if we assume standard scaling.
    EGO_X_PERCENT = 30
    
    cars_html = ""
    
    for i, row in enumerate(obs):
        # Skip if vehicle is inactive (all zeros)
        if np.allclose(row, 0.0) and i != 0:
            continue
            
        if i == 0:
            # Ego vehicle
            # Y is absolute lane
            left_pct = EGO_X_PERCENT
            top_pct = (ego_lane * LANE_HEIGHT_PERCENT) + (LANE_HEIGHT_PERCENT / 2)
            color = "#00E5FF" if vehicle_type == "R" else "#FF9100"
        else:
            # NPC vehicle
            rel_x = row[0]
            left_pct = EGO_X_PERCENT + (rel_x * 40)
            
            rel_y = row[1]
            # Estimate NPC absolute lane by adding relative Y (scaled) to ego lane
            npc_lane = ego_lane + (rel_y * 4)  # Rough approximation since rel_y is normalized by ~lane_width
            top_pct = (npc_lane * LANE_HEIGHT_PERCENT) + (LANE_HEIGHT_PERCENT / 2)
            color = "#757575"
            
        # Clamp visuals to container
        if left_pct < -10 or left_pct > 110:
            continue
        
        cars_html += CAR_TEMPLATE.format(
            left=left_pct,
            top=top_pct,
            svg_content=get_car_svg(color)
        )
        
    return SVG_TEMPLATE.format(cars_html=cars_html)
