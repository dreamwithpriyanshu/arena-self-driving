# UI Design System — Self-Driving Car Simulation

This document formalizes the aesthetic rules and visual language for the Streamlit MVP application.

## 1. Guiding Philosophy
**"A simulation lab tool, not a SaaS landing page."**
The UI should feel like a serious engineering dashboard. It must be highly responsive, focused on telemetry and visualization, and void of unnecessary visual fluff.

## 2. Color Palette
We avoid generic colors and glassmorphism in favor of a flat, high-contrast, dark-mode-first aesthetic.

- **Backgrounds**: `#1E1E24` (Deep asphalt gray).
- **Text/Labels**: `#E0E0E0` (Off-white for readability), `#9E9E9E` (Muted gray for secondary text).
- **Vehicle R (DQN) Accent**: `#00E5FF` (Cyan/Electric Blue). High visibility, represents the modern neural network approach.
- **Vehicle S (SARSA) Accent**: `#FF9100` (Safety Orange). Distinct from R, represents the classic tabular approach.
- **NPC Vehicles**: `#757575` (Neutral gray). They exist as obstacles, not the focus.

## 3. Typography
- **UI Text**: Streamlit's native sans-serif font (Inter/system default) to ensure fast loading and native feel.
- **Telemetry Readouts**: `monospace` for all rapidly updating numbers (speed, lane, step count) to prevent horizontal jitter during animations.

## 4. Visual Assets
All imagery is generated deterministically as SVG files to ensure infinite scaling and zero latency during live renders.

| Asset | Path | Usage |
|-------|------|-------|
| DQN Car Sprite | `assets/car_r.svg` | Representation of Vehicle R in Simulation & Live Tracking |
| SARSA Car Sprite | `assets/car_s.svg` | Representation of Vehicle S in Simulation & Live Tracking |
| NPC Car Sprite | `assets/car_npc.svg` | Representation of background traffic |
| Road Texture | `assets/road_texture.svg` | Background for the simulation viewport |

## 5. Animation & Charting
- **Animation**: The "Live Tracking" view utilizes a lightweight Python loop with `st.empty()` and `time.sleep()`. This avoids heavy 3D WebGL overhead while providing a responsive top-down view.
- **Charting Library**: **Altair** (`st.altair_chart`). Native to Streamlit, declarative, and renders cleanly in dark mode without the heavy interactive chrome of Plotly.
- **Chart Animation**: Performance charts use an **Incremental Update** model, redrawing every $N$ steps to create a live ticker effect without bogging down the browser.
