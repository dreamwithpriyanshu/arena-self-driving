# Self-Driving Car Simulation MVP

> An arrow-key driving simulator with one transparent, tabular SARSA policy
> trained from human demonstrations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)]()
[![HighwayEnv](https://img.shields.io/badge/sim-HighwayEnv-green.svg)]()

## What is this?

The project deliberately uses one non-neural approach: tabular SARSA. Its
Q-table is fast to train, easy to inspect, and avoids a neural-network stack.

The project has two runtime surfaces:

- **Native PyGame** handles arrow-key data collection, gameplay, and agent
  playback, including clickable in-window controls.
- **Streamlit** is a data and analytics dashboard. It does not run or render
  the simulation.

Native runtime examples are documented with screenshots in
[`tutorials/simulation_output.md`](tutorials/simulation_output.md).

## Project structure

```text
arena-self-driving/
├── app.py                         # Streamlit overview/dashboard entry point
├── pages/
│   ├── 1_Human_Demonstrations.py  # Dataset statistics and metadata
│   ├── 2_Performance_Analytics.py # Training-history charts
│   └── 3_Docs.py                  # Documentation viewer
├── src/
│   ├── envs/                      # HighwayEnv, actions, state builders
│   ├── agents/                    # DQN (R) and SARSA (S)
│   ├── human/                     # Keyboard recording workflow
│   ├── data/                      # Schemas, validation, loaders, dashboard data
│   ├── training/                  # Headless training orchestration
│   └── simulation/                # Shared environment lifecycle
├── scripts/                       # Native PyGame and headless CLI entry points
├── configs/                       # YAML environment defaults
├── data/                          # Human demonstrations and autonomous logs
├── artifacts/                     # Checkpoints and training history
├── tests/                         # pytest suite
├── docs/                          # Architecture and project notes
└── tutorials/                     # Commands and operating guides
```

## Setup

Prerequisites: Python 3.9–3.11 and pip.

```powershell
git clone https://github.com/dreamwithpriyanshu/arena-self-driving.git
cd arena-self-driving
python -m venv venv
venv\Scripts\activate
# Native PyGame simulator and trainer:
pip install -r requirements-native.txt
# Streamlit dashboard deployment uses requirements.txt automatically.
```

## Quick start

### Collect human demonstrations

Use the native PyGame window. The default profile is arrow keys; press `C` on
the setup screen to use WASD instead. `P` pauses without recording a step, and
`ESC` discards the active episode. The control file is
`configs/control_bindings.json` if you need to remap either profile.

```powershell
python scripts/play_human.py --vehicles-count 20 --duration 150
```

### Train the agents

```powershell
python scripts/train.py --episodes 50 --warm-start --seed 1000
```

This writes checkpoints and a timestamped training-history JSONL file under
`artifacts/`. Continue from checkpoints with `--resume`; for a no-update
comparison, use `--evaluation-only --resume --seed 2000`. Each R/S evaluation
episode receives the same seed.

### Watch agents natively

```powershell
python scripts/play_agent.py
```

Add `--train --save` to a playback script to update and persist the agents
during the native session.

The native windows show live step, action, reward, lane, speed, and terminal
status data. In multi-agent mode, R/DQN and S/SARSA are placed side-by-side in
adjacent lanes. `SPACE` pauses, `H` toggles telemetry, `S` saves checkpoints,
and `ESC` closes after a crash or timeout. Human mode samples held keys at
60 FPS and applies a discrete decision at 15 Hz. Its HUD shows both held input
and the action applied. Press `S` to save, or `X` to discard, only after a
recording has completed.

All native modes use the same controls: `+`/`-` change target speed, `/` cycles
City (12 m/s), Highway (18 m/s), and Express (24 m/s) pace presets, and the
HUD exposes matching clickable controls. NPC count and density are chosen on
the setup screen because they are created when the HighwayEnv episode resets.
In single-agent `--train` mode, `Q`/`E` (or `-EPS`/`+EPS`) decreases/increases
exploration epsilon without restarting the session.

Before either native session starts, the GUI provides controls for NPC vehicle
count, traffic density, target speed, and episode duration. Defaults are slower
than the original runtime: 18 m/s target speed and 15 decisions per second in
multi-agent playback.
In multi-agent mode, one crashed car no longer ends the run; both cars are
allowed to reach the crashed state.

### Open the data dashboard

```powershell
streamlit run app.py
```

The dashboard contains Overview, Human Demonstrations, Performance Analytics,
and Documentation pages. Performance Analytics can overlay selected timestamped
runs to compare R and S beyond the latest session. It reads files only; it
does not start HighwayEnv.

### Run tests

```powershell
pytest tests/ -v
```

## Evidence tracked by the project

| Evidence | Location | Produced by |
|---|---|---|
| Human demonstrations | `data/human_demonstrations/*.jsonl` | `play_human.py` |
| Agent checkpoints | `artifacts/checkpoints/` | `train.py` / playback `--save` |
| Training history | `artifacts/training_history_<run>.jsonl` | `train.py` |
| Run metadata | `artifacts/training_history_<run>.meta.json` | `train.py` |

The dashboard reports whatever is actually present. It does not manufacture
performance numbers or claim that a model is trained when its checkpoint is
missing.

## Documentation

- [Architecture](docs/architecture.md)
- [Training Guide](tutorials/training_guide.md)
- [Native Simulation Output](tutorials/simulation_output.md)
- [Command Reference](tutorials/commands.md)
- [Algorithm Notes](docs/algorithm_notes.md)
- [UI Design System](docs/ui_design_system.md)
- [Security Notes](docs/security_notes.md)
- [7-Day Human Training Plan](docs/human_training_7_day_plan.md)
- [Project Timeline](docs/timeline.md)

This project is for educational purposes.
