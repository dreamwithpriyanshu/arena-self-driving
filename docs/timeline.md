# Project Status

## Current product

Arena Self-Driving is a local educational highway-driving project with one
transparent tabular SARSA policy. It combines arrow-key human demonstrations,
SARSA warm start and training, native policy playback, and a local web control
surface for metrics and evidence review.

## Completed capabilities

- HighwayEnv simulation with a five-action discrete control contract.
- Native PyGame human recorder and SARSA playback windows.
- Validated JSONL demonstrations and warm-start ingestion.
- One tabular SARSA Q-table, checkpoint persistence, and greedy evaluation.
- Robust state construction for full Kinematics matrices and supported ego-only
  observations, with warnings when neighbour rows must be zero-padded.
- FastAPI server bound to loopback, fixed subprocess launch arguments, and
  WebSocket metric streaming.
- Browser controls for human drive, agent playback, SARSA training, stop,
  saved-run analytics, demonstration summary, and project documentation.

## Evidence locations

- Demonstrations: `data/human_demonstrations/`
- Latest SARSA table: `artifacts/checkpoints/sarsa_q_table.npy`
- Browser runs: `artifacts/runs/<browser-run-id>/`
- Older direct-script histories: `artifacts/training_history_*.jsonl`

See the README for the end-to-end workflow and `backend_notes.md` for API and
local-security details.
