# Architecture — Self-Driving Car Simulation MVP

## Runtime split

The project has two deliberate front ends:

```text
                         reads
  native PyGame runners ────────┐
  headless trainer ─────────────┼──> data/ and artifacts/
                                │
  Streamlit dashboard <────────┘
       (overview, dataset, analytics, docs only)

  envs ──> simulation ──> human recorder ──> data
    │          │               │
    └────────> agents <──────── training ──> artifacts
```

Streamlit does not create, step, or render HighwayEnv. Its job is to inspect
validated files produced by the native and headless runtimes. This keeps the
web UI responsive while PyGame owns the 60 FPS interaction loop.

## Layers and responsibilities

### `src/envs/` — environment contract

Owns HighwayEnv configuration, the shared action set, and continuous and
discrete state builders. It does not know about Streamlit or agents.

### `src/simulation/` — environment lifecycle

`env_manager.py` wraps reset, step, state construction, reward accumulation,
and close operations. It is used by native PyGame scripts, the human recorder,
and the trainer. It is not a Streamlit façade.

### `src/agents/` — learning algorithms

- Vehicle R: DQN with a PyTorch network and replay buffer.
- Vehicle S: tabular SARSA with a NumPy Q-table.

Both expose action, update, checkpoint, and evaluation-mode operations.

### `src/human/` — native human recorder

`EpisodeManager` combines `EnvManager`, `KeyboardController`, and the data
recorder. `scripts/play_human.py` captures keyboard input in a native PyGame
window and writes validated JSONL demonstrations.

### `src/training/` — headless orchestration

`TrainingOrchestrator` loads demonstrations for warm starts, runs episodes for
R and S, updates the agents, and saves checkpoints. `scripts/train.py` writes
per-run NDJSON history plus metadata under `artifacts/`.

### `src/data/` — persisted evidence

Schemas, validation, recording, loading, dataset summaries, and dashboard
history helpers live here. Human demonstrations and autonomous logs remain
separate from model checkpoints and training history.

### `scripts/` — operational entry points

| Script | Role |
|---|---|
| `play_human.py` | Native human demonstration recording |
| `play_agent.py` | Native single-agent evaluation or live training |
| `play_multi_agent.py` | Native DQN/SARSA side-by-side evaluation or training |
| `train.py` | Headless batch training and history generation |

### Streamlit UI

| File | Role |
|---|---|
| `app.py` | Overview, evidence summary, run selector, native workflow guide |
| `pages/1_Human_Demonstrations.py` | Dataset counts, vehicle split, episode metadata |
| `pages/2_Performance_Analytics.py` | Altair comparison charts and raw training history |
| `pages/3_Docs.py` | Renders project documentation |

The pages import data loaders only. They do not import `envs`, `agents`,
`training`, or `simulation`, and they never start a simulation.

## Data flow

1. `play_human.py` records state/action/reward transitions to
   `data/human_demonstrations/*.jsonl`.
2. `train.py --warm-start` loads those demonstrations into DQN replay and
   SARSA updates, then trains headlessly.
3. Training writes checkpoints to `artifacts/checkpoints/` and history to
   `artifacts/training_history_<run>.jsonl` plus `.meta.json`.
4. The dashboard reads those files and plots only what exists on disk.
5. `play_agent.py` or `play_multi_agent.py` provides native visual evaluation.

## Configuration and boundaries

Environment defaults are in `configs/default_env.yaml`; CLI flags override
traffic count, density, duration, and training settings. Data paths are
project-relative. JSONL files are validated before training, and generated
datasets, logs, checkpoints, and histories are ignored by Git.
