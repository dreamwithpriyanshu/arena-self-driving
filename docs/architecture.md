# Architecture — Self-Driving Car Simulation MVP

## Overview

The system is organised into five layers with **one-directional dependencies**.
Each layer may only import from layers to its left.  Streamlit pages may only
call through the `simulation` façade — never into `envs`, `agents`, or
`training` directly.

```
envs  →  agents  →  training  →  simulation  →  UI (pages)
                data  ↗
```

---

## Layer descriptions

### 1. `src/envs/` — Environment

Owns the HighwayEnv configuration, the shared action set, and the state
builder (both continuous and discrete representations).

| Module | Responsibility |
|--------|---------------|
| `actions.py` | `Action` enum, `NUM_ACTIONS`, validation, keyboard map |
| `highway_factory.py` | Creates a configured `gymnasium.Env` from YAML |
| `state_builder.py` | `build_raw_state()` → flat vector (DQN), `build_discrete_state_with_lane()` → int (SARSA) |

**Knows nothing about** agents, training, Streamlit, or file I/O beyond
reading `configs/default_env.yaml`.

### 2. `src/agents/` — Agents

Contains two independent agent classes:

- **DQN** (vehicle R) — neural-network Q-learner using a replay buffer
- **SARSA** (vehicle S) — tabular on-policy learner

Both implement the same interface: `act()`, `update()`, `save()`, `load()`,
`set_eval_mode()`.  Neither agent module imports the other.

### 3. `src/data/` — Data

Recording, validation, and loading of human demonstrations and autonomous
logs.  Pure functions where possible.  No agent or training logic.

### 4. `src/human/` — Human Recorder

Keyboard-driven control, episode management, save/discard workflow.
Depends on `envs` and `data` only.

### 5. `src/training/` — Training

Orchestrates envs + agents + data.  The **only** layer that knows about
both R (DQN) and S (SARSA) at the same time.

### 6. `src/evaluation/` — Evaluation

Fixed-seed evaluation runs, metric collection, comparison logic.
Depends on `envs`, `agents`, and `data`.

### 7. `src/simulation/` — Simulation Façade

A thin wrapper that Streamlit pages call.  Starts/steps the env, exposes
current state for rendering, exposes metrics.

| Module | Responsibility |
|--------|---------------|
| `env_manager.py` | `EnvManager` class — reset, step, close |
| `training_facade.py` | `TrainingFacade` class — exposes UI-safe orchestrator methods |


**Pages never import** from `envs`, `agents`, or `training` directly.

### 8. `pages/` — Streamlit UI

Rendering and interaction only.  No training logic, no file I/O beyond
what `simulation`/`data` expose.

---

## Key data types crossing layer boundaries

| Type | Defined in | Used by |
|------|-----------|---------|
| `Action` (IntEnum) | `envs.actions` | all layers |
| `StepResult` (dataclass) | `simulation.env_manager` | pages, training |
| Raw observation `np.ndarray(V,F)` | HighwayEnv | envs, simulation |
| Discrete state `int` | `envs.state_builder` | agents (SARSA) |
| Flat state `np.ndarray(V*F,)` | `envs.state_builder` | agents (DQN) |

---

## Configuration

All environment parameters live in `configs/default_env.yaml`. All paths are resolved relative to the project root dynamically.

---

## Security boundaries

- File I/O is confined to `data/` and `artifacts/` within the project root.
- No `eval`/`exec`/`pickle.load` on untrusted files.
- All user inputs are validated before use.
- See `docs/security_notes.md` for details.

---

*Last updated: Build Step 5 — QA & Handoff*
