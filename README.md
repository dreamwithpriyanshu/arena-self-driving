# Self-Driving Car Simulation MVP

> A comparative study of DQN versus SARSA for autonomous highway driving,
> trained from shared human demonstrations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)]()
[![HighwayEnv](https://img.shields.io/badge/sim-HighwayEnv-green.svg)]()

## What is this?

Vehicle R uses DQN, an off-policy neural-network Q-learner. Vehicle S uses
tabular SARSA, an on-policy learner. Both receive the same human-driving data;
the algorithm is the controlled difference.

The project has two runtime surfaces:

- **Native PyGame** handles human data collection, gameplay, agent playback,
  and the side-by-side DQN/SARSA view.
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
pip install -r requirements.txt
```

## Quick start

### Collect human demonstrations

Use the native PyGame window. ENTER starts; arrow keys drive; ESC discards the
active episode.

```powershell
python scripts/play_human.py R
python scripts/play_human.py S --vehicles-count 20 --duration 150
```

### Train the agents

```powershell
python scripts/train.py --agent BOTH --episodes 50 --warm-start --history-mode per-run
```

This writes checkpoints and a timestamped training-history JSONL file under
`artifacts/`.

### Watch agents natively

```powershell
python scripts/play_agent.py R
python scripts/play_agent.py S
python scripts/play_multi_agent.py --duration 300
```

Add `--train --save` to a playback script to update and persist the agents
during the native session.

### Open the data dashboard

```powershell
streamlit run app.py
```

The dashboard contains Overview, Human Demonstrations, Performance Analytics,
and Documentation pages. It reads files only; it does not start HighwayEnv.

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

This project is for educational purposes.
