# Project Timeline and Final Submission State

## Phase 1 — Foundation

The project began with a configurable HighwayEnv highway, a shared action
contract, compact state construction, an environment lifecycle manager, JSONL
episode recording, validation, loading, and a native PyGame human driver.

## Phase 2 — Native driving and SARSA

The browser simulation was replaced by native PyGame windows so keyboard input,
rendering, and driving controls stay local. The project was simplified to one
non-neural learner: tabular SARSA (`S`). Arrow keys remain the only driving
controls. The training environment uses responsive five-Hz policy decisions,
moderate default traffic, reset spacing, a stronger collision penalty, and a
small lane-change cost.

## Phase 3 — Robust state construction

State builder accepts either a full `(V, F)` Kinematics matrix or a 1-D ego
vector. The 1-D case is deliberately accepted because supported environment
configurations can emit it; the ego features are retained and neighbour rows
are zero-padded with warnings. Callers requiring neighbour information must
request full Kinematics observations or use the best-effort reconstruction
helper. The reconstruction helper also warns if it has to zero-pad missing
vehicle rows.

## Phase 4 — Local web control surface

Streamlit was retired in favour of FastAPI and a lightweight browser frontend.
The server binds to loopback by default, validates bounded SARSA
hyperparameters, launches the existing trainer with a fixed `shell=False`
argument list, and streams JSONL metrics to the browser. The frontend provides
start, stop, live status, live reward charting, and saved-run analytics without
a Node.js build chain.

## Submission checklist

- Arrow-key native GUI demonstration recorder (`scripts/play_human.py`)
- GUI SARSA playback with HUD and clickable controls (`scripts/play_agent.py`)
- Headless reproducible SARSA trainer and Q-table checkpoint (`scripts/train.py`)
- FastAPI training control and analytics (`backend/`, `frontend/`)
- Validated JSONL evidence and automated environment, data, training, and API checks
- Full documentation suite under `docs/`, including backend operating notes
