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

# Record arrow-key demonstrations in the native GUI
python scripts/play_human.py

# Train SARSA from recorded demonstrations
python scripts/train.py --episodes 50 --warm-start --seed 1000

# Watch the trained policy in the native GUI
python scripts/play_agent.py

# Start the local training control surface, then open http://127.0.0.1:8000
python -m backend.main
```

The browser frontend requires no Node.js or bundler; FastAPI serves the plain
HTML, CSS, and JavaScript files in `frontend/`.

## Evidence

| Item | Location |
|---|---|
| Human demonstrations | `data/human_demonstrations/*.jsonl` |
| SARSA checkpoint | `artifacts/checkpoints/sarsa_q_table.npy` |
| Training history | `artifacts/training_history_<run>.jsonl` |
| Run metadata | `artifacts/training_history_<run>.meta.json` |

See [Architecture](docs/architecture.md), [Commands](docs/commands.md), and
[Timeline](docs/timeline.md) for the submission summary.
