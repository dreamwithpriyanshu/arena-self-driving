# Project Timeline and Final Submission State

## Phase 1 — Foundation

The project began with a configurable HighwayEnv highway, a shared action
contract, compact state construction, and an environment lifecycle manager.
JSONL episode recording, validation, loading, and a native PyGame human driver
were then added.

## Phase 2 — GUI and evidence workflow

The browser simulation was replaced by native PyGame windows so keyboard input,
rendering, and real-time controls run locally. Streamlit was retained as an
evidence dashboard for demonstrations, metrics, and documentation.

## Phase 3 — Final consolidation

For the final submission, the experiment was simplified to one non-neural
approach: tabular SARSA. Arrow keys are the only driving controls. The active
trainer, native viewer, dashboard, tests, and documentation now describe that
single workflow. The training environment was improved with responsive
five-Hz policy decisions, moderate default traffic, reset spacing, a stronger
collision penalty, and a small lane-change cost.

## Phase 4 — Deployment fixes

The Streamlit deployment was separated from native dependencies: Cloud installs
dashboard-only requirements, while desktop simulation uses
`requirements-native.txt`. Altair 6 is required because it supports the Python
3.14 environment used by Streamlit Cloud.

## Phase 5 — 1-D observation robustness and documentation

State builder (`src/envs/state_builder.py`) was hardened for 1-D ego-only
observation vectors. The builder now accepts either a full `(V, F)` matrix or a
1-D `(F,)` ego vector and expands the latter into a zero-padded `(V, F)` matrix.
A best-effort `build_raw_state_from_env` reconstruction helper was added for
cases where the environment exposes vehicle objects directly.

Tests in `tests/test_envs.py` were updated to assert the new expansion behaviour
instead of expecting a `ValueError`. This relaxation is deliberate: an
ego-only observation is a supported shape emitted by some environment
configurations, so rejecting it would make training brittle. The expansion
preserves the ego features and explicitly warns that neighbour rows are
zero-padded; callers that require neighbour data must request full Kinematics
observations or use the best-effort environment reconstruction helper. The
checkpoint round-trip test was fixed to use a project-local directory (avoiding
Windows `tmp_path` permission errors).

Stale DQN/two-agent references in docstrings and comments were cleaned out
across `state_builder.py`, `actions.py`, and `episode_manager.py`.

Documentation was expanded: `docs/commands.md` (command reference),
`docs/architecture.md`, `docs/algorithm_notes.md`, `docs/security_notes.md`,
`docs/ui_design_system.md`, and this timeline were refreshed for submission.
The Streamlit Documentation page (`pages/3_Docs.py`) displays all six docs.

## Submission checklist

- Arrow-key native GUI demonstration recorder (`scripts/play_human.py`)
- GUI SARSA playback with HUD and clickable controls (`scripts/play_agent.py`)
- Headless reproducible SARSA trainer and Q-table checkpoint (`scripts/train.py`)
- Validated JSONL evidence and Streamlit analytics dashboard (`app.py`, `pages/`)
- Deployment-safe dependency split (`requirements.txt` vs `requirements-native.txt`)
- Automated environment, data, training, and UI checks (16 tests, all passing)
- State builder robust to both `(V, F)` and 1-D ego-only observations
- All stale DQN/multi-agent references removed from codebase
- Full documentation suite under `docs/` visible in the Streamlit Docs tab
