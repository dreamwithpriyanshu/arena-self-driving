# Arena Self-Driving

Arrow-key highway driving demonstrations train one transparent, non-neural
tabular SARSA policy. This is an educational simulation, not a real driving
system.

## Final design

- **Controls:** Arrow Left/Right change lanes; Arrow Up accelerates; Arrow Down
  slows; Space is idle. These are the only driving bindings.
- **Native GUI:** PyGame owns driving, playback, pause, HUD, clickable speed
  controls, traffic setup, and save/discard actions.
- **Learning:** One tabular SARSA Q-table. There is no DQN, PyTorch, GPU, or
  multi-agent comparison path.
- **Control surface:** FastAPI serves a local browser frontend for training
  controls, live metrics, and saved-run analytics. It binds to loopback only.

## Install and run locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements-native.txt

# Start the local training control surface, then open http://127.0.0.1:8000
python -m backend.main
```

The browser frontend requires no Node.js or bundler; FastAPI serves the plain
HTML, CSS, and JavaScript files in `frontend/`.

## Browser workflow

1. Open **Human demonstration** to launch the native PyGame driving window.
   Use arrow keys to drive and save an episode when finished.
2. Back in the browser, enable **Warm-start from saved demonstrations** if the
   new training run should learn from those recordings.
3. Configure bounded SARSA hyperparameters and start training. Four live
   charts show reward, survival steps, exploration epsilon, and TD error.
4. Open **SARSA playback** to launch the native viewer for the saved checkpoint.
5. Select a saved run to revisit its analytics, or use **Documentation** to
   read the architecture, algorithm, controls, security, and backend notes.

The human-driving and agent-play windows remain native PyGame applications;
the browser can launch them locally but does not embed their desktop windows.

## Evidence

| Item | Location |
|---|---|
| Human demonstrations | `data/human_demonstrations/*.jsonl` |
| SARSA checkpoint | `artifacts/checkpoints/sarsa_q_table.npy` |
| Browser-started run history | `artifacts/runs/<browser-run-id>/training_history_<run>.jsonl` |
| Browser-started run metadata | `artifacts/runs/<browser-run-id>/training_history_<run>.meta.json` |
| Earlier command-line history | `artifacts/training_history_<run>.jsonl` |

`run-id` is an opaque identifier created by the backend. It keeps each
browser-started run in its own folder, so the live websocket and saved-run
browser can identify its history without accepting a client-provided path.

See [Architecture](docs/architecture.md), [Commands](docs/commands.md), and
[Timeline](docs/timeline.md) for the submission summary.
