# Arena Self-Driving: Project Documentation

## 1. Project overview

Arena Self-Driving is a local educational highway-driving workbench built
around one transparent, tabular SARSA policy. It combines:

- A HighwayEnv traffic simulation
- A native PyGame human-driving recorder
- A native PyGame SARSA playback viewer
- A headless SARSA trainer
- A FastAPI browser control surface
- Saved demonstrations, checkpoints, run history, and evaluation reports

The project is intended for learning and experimentation. It is not a real
vehicle controller or a safety-certified driving system.

## 2. Main workflow

```text
Human driving (optional)
        |
        v
Saved demonstrations
        |
        v
SARSA training with --warm-start
        |
        v
Saved Q-table checkpoint
        |
        +--> Native agent playback
        |
        +--> Greedy evaluation and metrics
```

The recommended workflow is:

```powershell
.\venv\Scripts\Activate.ps1
python -m backend.main
```

Open `http://127.0.0.1:8000` to use the browser workbench.

## 3. Installation

Run these commands from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m backend.main
```

The frontend is served directly by FastAPI. Node.js and a frontend bundler are
not required.

## 4. Human demonstrations

Start the recorder:

```powershell
python scripts/play_human.py
```

The orange car is the human-controlled vehicle. Blue cars are NPC traffic.

During the episode:

- Left/Right change lanes
- Up/Down apply faster/slower driving actions
- `P` pauses
- `H` toggles the HUD
- `[` and `]` reduce or increase duration
- `/` changes the road mode
- `Esc` discards the active episode

When the episode ends:

- Press `S` to save it
- Press `X` to discard it
- Press `Esc` to close without saving

Saved demonstrations are written to:

```text
data/human_demonstrations/*.jsonl
```

Saving is required before a demonstration can be used for warm-start training.

## 5. Training

### Fresh training

```powershell
python scripts/train.py --episodes 100 --seed 2000
```

This starts with a zero-initialised Q-table. It does not delete demonstration
files or previous history, but it replaces the runtime checkpoint when the new
training run saves.

### Continue training

```powershell
python scripts/train.py --episodes 100 --resume --seed 2000
```

`--resume` loads the existing Q-table and continues learning from it. This is
the normal command for improving an existing model.

### Use human demonstrations

```powershell
python scripts/train.py --episodes 100 --resume --warm-start --seed 2000
```

`--warm-start` reads saved JSONL demonstrations and applies SARSA updates before
the autonomous episodes. It does not permanently copy the human policy.

### Important training options

| Option | Meaning |
|---|---|
| `--episodes` | Number of episodes to run |
| `--resume` | Load the existing Q-table |
| `--warm-start` | Use saved human demonstrations first |
| `--seed` | Make random experience repeatable |
| `--duration` | Maximum decisions per episode |
| `--vehicles-count` | Number of NPC vehicles |
| `--vehicles-density` | Traffic density |
| `--evaluation-only` | Run without learning updates |
| `--greedy` | Disable exploration |
| `--lr` | SARSA learning rate |
| `--gamma` | Future reward discount |
| `--epsilon-start` | Initial exploration |
| `--epsilon-end` | Minimum exploration |
| `--epsilon-decay` | Exploration decay rate |

## 6. Environment step

One training decision follows this sequence:

1. The environment resets the road and vehicles.
2. The observation is converted to a discrete SARSA state.
3. The policy chooses one of five actions:
   `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, or `SLOWER`.
4. HighwayEnv advances the ego vehicle and NPC traffic.
5. The environment returns the next observation, reward, and termination flags.
6. SARSA updates `Q(state, action)`.
7. Episode metrics are written to JSONL.
8. The next decision begins until the episode ends.

An episode ends after a collision, truncation, or its configured limit.

The Q-table update is:

```text
Q(s,a) += learning_rate *
          (reward + gamma * Q(next_state,next_action) - Q(s,a))
```

## 7. Model, checkpoint, and data behavior

The project has one model: a tabular SARSA Q-table.

```text
artifacts/checkpoints/sarsa_q_table.npy
```

Without `--resume`, the old Q-table is not loaded. The new run starts from
zero-valued Q-values. Once the new run saves, the runtime checkpoint is
replaced, so the old learned progress is lost from that path.

The following are not deleted:

- `data/human_demonstrations/`
- Previous training histories and metadata
- `published/`
- Source code and configuration

Back up a checkpoint before a fresh run:

```powershell
Copy-Item artifacts\checkpoints\sarsa_q_table.npy `
  artifacts\checkpoints\sarsa_q_table.backup.npy
```

For an isolated experiment:

```powershell
$env:ARENA_STORAGE_DIR = "D:\arena-experiment-2"
python scripts/train.py --episodes 100 --seed 2000
Remove-Item Env:ARENA_STORAGE_DIR
```

Changing only `--seed` never deletes or resets a model.

## 8. Agent playback

Start greedy playback:

```powershell
python scripts/play_agent.py
```

The agent viewer loads the saved checkpoint and drives autonomously. It supports
pause, target speed, road mode, duration, and HUD controls.

To train while watching:

```powershell
python scripts/play_agent.py --train --save
```

This is interactive training and should be used carefully because `--save`
writes the resulting checkpoint on exit.

## 9. Evaluation

Run repeatable greedy evaluation:

```powershell
python scripts/evaluate_model.py --episodes 10 --steps 300 --seed 1000
```

Evaluation does not update the Q-table. It appends:

- Average reward
- Average survival steps
- Collision-free rate

to `accuracy.md`.

## 10. Publishing

Publish the current local checkpoint explicitly:

```powershell
python scripts/publish_model.py
git add published accuracy.md
git commit -m "Publish improved SARSA baseline"
git push
```

Publishing copies files into `published/`; it does not commit or push by
itself.

## 11. Storage layout

| Path | Purpose |
|---|---|
| `data/human_demonstrations/` | Saved human episodes |
| `artifacts/checkpoints/` | Runtime SARSA Q-table |
| `artifacts/runs/` | Browser run history and metadata |
| `artifacts/training_history_*.jsonl` | Command-line history |
| `published/checkpoints/` | GitHub-shared baseline |
| `accuracy.md` | Evaluation timeline |
| `frontend/` | Browser UI |
| `backend/` | FastAPI server |
| `src/` | Environment, agent, data, and training code |
| `scripts/` | User-facing commands |
| `tests/` | Automated tests |

## 12. Validation

```powershell
python -m pytest
python -m compileall -q backend scripts src tests
```

These commands validate the project. They do not train or overwrite the model.

## 13. Script summary

| Script | Function |
|---|---|
| `backend.main` | Starts the local browser workbench |
| `play_human.py` | Records and saves human demonstrations |
| `train.py` | Trains, resumes, warm-starts, or evaluates SARSA |
| `play_agent.py` | Displays autonomous SARSA playback |
| `evaluate_model.py` | Measures the saved model without learning |
| `publish_model.py` | Copies the runtime model into `published/` |

## 14. Safety and scope

The project is local-first and educational. It uses a simulated road and a
small discrete state/action representation. Results depend on environment
configuration, random seed, checkpoint state, demonstrations, and training
duration. A high reward in simulation does not imply real-world driving safety.

---

Made with heart by Priyanshu ❤️
