# Dashboard UI Design System

The Streamlit app is a restrained evidence dashboard, not a simulation
frontend. It makes recorded data and comparisons easy to inspect without
pretending that the browser is driving the environment.

## Palette

| Token | Value | Use |
|---|---|---|
| Canvas | `#1E1E24` | Main background |
| Panel | `#282830` | Sidebar and secondary surfaces |
| Text | `#E0E0E0` | Primary text |
| R / DQN | `#00E5FF` | Vehicle R charts and labels |
| S / SARSA | `#FF9100` | Vehicle S charts and labels |

The palette is configured in `.streamlit/config.toml` with `base = "dark"` so
Streamlit's sidebar, controls, and native chrome use the same dark theme.

## Page set

- **Overview** (`app.py`): what the dashboard reads and where native commands
  fit in the workflow.
- **Human Demonstrations**: dataset totals, vehicle split, and episode metadata.
- **Performance Analytics**: incremental Altair charts from actual training
  history files, followed by the raw records.
- **Documentation**: project guidance rendered from `docs/`.

## Layout rules

- Prefer readable tables and charts over decorative metric-card grids.
- Keep one clear heading and one explanatory caption per section.
- Use R/S colors consistently; never use color as the only explanation.
- Show the source filename when displaying training history.
- Empty states must tell the user which native command produces the missing
  evidence.
- Do not add simulation controls, live rendering, SVG car drawings, or fake
  result values to Streamlit.
