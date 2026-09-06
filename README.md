# Arena Self-Driving: SARSA Workbench

Arena Self-Driving is a local educational highway-driving project. Arrow-key
demonstrations warm-start one transparent, non-neural tabular SARSA policy.
It is not a real driving system.

## What is included

- **One model:** `S` is the only model label and always means tabular SARSA.
- **Human demonstrations:** a native PyGame window records arrow-key episodes.
- **Agent playback:** a native PyGame window visualises the saved SARSA table.
- **Training workbench:** FastAPI serves a local browser UI for controls, live
  metrics, saved-run charts, documentation, and native-window launch buttons.
- **Local-only server:** the workbench binds to `127.0.0.1`, not a network
  interface.

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

## Deploy to Render

Use the included [`render.yaml`](render.yaml) as a Blueprint. It installs the
Python dependencies, binds FastAPI to Render's public port, and mounts a
persistent disk at `/var/data`. `ARENA_STORAGE_DIR=/var/data` keeps human
demonstrations, checkpoints, and all run history across deploys and restarts.
Persistent disks require a paid Render instance and are attached to one service
instance; do not scale this service horizontally unless storage is moved to
object storage or a database.

Render is a headless server, so a human cannot record keyboard demonstrations
there. The **SARSA playback** button uses a headless evaluation run instead of
opening PyGame, and saves its metrics in run history. Training, warm-starting
from stored demonstration files, live metrics, checkpoints, and saved history
work through the browser API. Use the local workflow for arrow-key recording,
then copy validated JSONL files into the mounted storage when needed.

## Sharing an improved model through GitHub

Training files under `artifacts/` and `data/` are intentionally ignored by Git,
so pushing the repository alone does **not** share your learned model. To
publish the current local checkpoint and latest history:

```powershell
python scripts/publish_model.py
git add published
git commit -m "Publish improved SARSA baseline"
git push
```

The tracked `published/` baseline is automatically copied into writable storage
on a fresh install or Render disk. Existing storage is never overwritten, so a
user who has already trained keeps their newer checkpoint and history. Each
published update must be committed and pushed explicitly.

Browser training continues from the saved checkpoint by default, so repeated
Render runs improve the existing model instead of resetting it. Disable the
resume checkbox only when an intentional fresh experiment is needed.

### What appears in GitHub after Render training?

Nothing is pushed automatically. Render's persistent disk is separate from the
GitHub repository, and training updates the checkpoint and history on that disk
only. To put a trained Render model into GitHub, download the checkpoint/history
from the service, place them under `artifacts/`, run `python
scripts/publish_model.py`, then commit and push `published/`. Alternatively,
keep the trained model on Render and treat GitHub as the source code/baseline
repository.

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

## SARSA training data

The environment has five actions: lane left, idle, lane right, faster, and
slower. SARSA learns a Q-value for a discretised traffic state and action. Its
live charts show total reward, survival steps, exploration epsilon, and average
TD error. See [Algorithm Notes](docs/algorithm_notes.md) for the update rule
and state representation.

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

## Documentation

The workbench exposes all project docs under **Documentation**. They are also
available directly:

- [Architecture](docs/architecture.md)
- [SARSA algorithm](docs/algorithm_notes.md)
- [Commands and workflow](docs/commands.md)
- [FastAPI control surface](docs/backend_notes.md)
- [Security and data notes](docs/security_notes.md)
- [UI design](docs/ui_design_system.md)
- [Project status](docs/timeline.md)
