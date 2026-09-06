# Architecture

Arena Self-Driving has one learning path and two local presentation surfaces.

```text
Human drive button -> native PyGame recorder -> validated JSONL demonstrations
                                                |
Browser training form -> FastAPI -> scripts/train.py -> SARSA Q-table
                           |                         -> per-run JSONL metrics
                           +-> WebSocket -> live browser charts

Agent play button -> native PyGame viewer -> saved SARSA Q-table
```

## Responsibilities

| Area | Responsibility |
|---|---|
| `src/envs/` | HighwayEnv configuration, action validation, continuous-to-discrete state conversion. |
| `src/simulation/` | Environment lifecycle, reset, step, state, and metrics facade. |
| `src/human/` | Arrow-key capture and human episode lifecycle. |
| `src/data/` | JSONL schemas, recorder, validator, demonstration loader, and summaries. |
| `src/agents/sarsa.py` | The sole tabular SARSA policy and checkpoint persistence. |
| `src/training/` | Warm start, training/evaluation episode orchestration. |
| `scripts/` | Headless trainer and native PyGame entry points. |
| `backend/` | Local-only API, subprocess ownership, live metric streaming, static frontend hosting. |
| `frontend/` | No-build HTML, CSS, and JavaScript dashboard. |

## Local UI model

The browser is the control surface for demonstrations, playback, training,
analytics, and project documentation. Human driving and policy playback still
run in native PyGame windows because they require desktop keyboard input and
rendering. The browser launches those fixed local scripts; it does not embed a
simulator window.

## Evidence model

Human demonstrations are stored in `data/human_demonstrations/`. Browser
training requests receive a backend-generated ID and write under
`artifacts/runs/<browser-run-id>/`. Each directory contains a timestamped JSONL
history file and matching metadata file. The API also reads compatible older
history directly under `artifacts/`.
