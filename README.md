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
pip install -r requirements.txt

# Start the local training control surface, then open http://127.0.0.1:8000
python -m backend.main
```

The browser frontend requires no Node.js or bundler; FastAPI serves the plain
HTML, CSS, and JavaScript files in `frontend/`.

## Human demonstrations and warm-start training

Human demonstrations are optional, but they must be saved before training can
use them. The reliable workflow is:

1. Start the workbench with `python -m backend.main`, then open
   `http://127.0.0.1:8000`, or run `python scripts/play_human.py` directly.
2. In the PyGame window, choose the `S` model if prompted. Use the arrow keys:
   Left/Right change lane, Up accelerates, and Down slows down.
3. Drive until the episode ends. At the completion screen press **S** to save
   it, **X** to discard it, or **Esc** to close without saving.
4. Confirm a JSONL file exists under
   `data/human_demonstrations/`. The browser also shows the saved episode and
   transition count.
5. Start training with **Warm-start from saved demonstrations** enabled, or run:

   ```powershell
   python scripts/train.py --episodes 100 --resume --warm-start --seed 2000
   ```

Warm-start reads the saved transitions and applies SARSA updates before the
new autonomous episodes. It does not copy a human policy permanently and it
does not delete or modify the demonstration files. Without `--warm-start`,
demonstrations remain stored but are not used by that training run.

## Models, checkpoints, and seeds

This project has one model: a tabular SARSA Q-table stored at
`artifacts/checkpoints/sarsa_q_table.npy`. A “new model” means a fresh
zero-initialised Q-table for a training run, not a new model type.

- `python scripts/train.py --episodes 50` starts from a fresh table in memory
  and writes the resulting table to the checkpoint at the end. It does not
  delete demonstrations, old history, or published files; the single runtime
  checkpoint is replaced by the new result.
- `python scripts/train.py --episodes 50 --resume` loads the existing
  checkpoint and improves it.
- `--seed` controls repeatability: it seeds Python, NumPy, and episode seeds.
  A new seed does **not** create a new model, clear data, or reset the table.
  With `--resume`, it simply changes the random experience used while
  continuing the same table.
- The browser’s **Continue from the saved SARSA checkpoint** checkbox is
  equivalent to `--resume` and is enabled by default.

### What happens when `--resume` is not used?

Without `--resume`, `train.py` does **not** delete the whole project or clear
all saved data. It creates a fresh Q-table in memory, trains it, and writes the
result to the same runtime checkpoint path:

```text
artifacts/checkpoints/sarsa_q_table.npy
```

That means the old runtime Q-table is replaced when the new run checkpoints
(every `--save-freq` episodes and again at shutdown). The following data are
not deleted:

- Human demonstrations in `data/human_demonstrations/`
- Previous training history and metadata in `artifacts/`
- The published baseline in `published/`
- Source code and configuration

The old runtime table is not recoverable from that path after it is replaced,
so make a copy before starting a fresh experiment if you want to keep it.
Training history records the new run separately, but it does not contain a full
copy of the old Q-table.

Safe ways to create an independent experiment:

```powershell
# Continue the current model (safest default)
python scripts/train.py --episodes 100 --resume --seed 2000

# Start fresh in a separate storage directory
$env:ARENA_STORAGE_DIR = "D:\arena-experiment-2"
python scripts/train.py --episodes 100 --seed 2000
Remove-Item Env:ARENA_STORAGE_DIR
```

Changing `--seed` alone never resets, deletes, or creates a separate model.
Use `--resume` to improve the current table, or omit it only when you
intentionally want a fresh table.

## What the scripts do

| Script | Purpose | Reads | Writes or changes |
|---|---|---|---|
| `python -m backend.main` | Starts the local FastAPI browser workbench | Project code and docs | Starts a local server; browser actions launch the scripts below |
| `scripts/play_human.py` | Records arrow-key driving | Environment/configuration | Saves accepted episodes to `data/human_demonstrations/*.jsonl` |
| `scripts/train.py` | Trains or evaluates SARSA episodes | Optional checkpoint and optional demonstrations | Writes `artifacts/checkpoints/sarsa_q_table.npy` and run history/metadata |
| `scripts/play_agent.py` | Opens interactive native agent playback | Checkpoint | May write a checkpoint only with its training/save options |
| `scripts/evaluate_model.py` | Runs greedy, read-only evaluation | Checkpoint | Appends summary metrics to `accuracy.md`; does not train |
| `scripts/publish_model.py` | Copies the runtime checkpoint for GitHub | Runtime checkpoint and latest history | Updates `published/`; it does not commit or push |

`python -m pytest` runs the automated tests. `python -m compileall -q
backend scripts src tests` checks that Python files compile; neither command
trains or changes the model.

## Deploy to Render

For the free plan, follow [Free Render Web Service](docs/render_free_web_service.md).
It uses the headless requirements, Python 3.13.7, Render's `$PORT`, and temporary
`/tmp/arena-data` storage. Render is headless: human keyboard demonstrations
remain local-only, while training and SARSA evaluation run through the browser.

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

The tracked `published/` baseline checkpoint is automatically copied into
writable storage on a fresh install or Render instance. Existing storage is never overwritten, so a
user who has already trained keeps their newer checkpoint and history. Each
published update must be committed and pushed explicitly.

Browser training continues from the saved checkpoint by default, so repeated
Render runs improve the existing model instead of resetting it. Disable the
resume checkbox only when an intentional fresh experiment is needed.

## Measuring model improvement

SARSA does not have classification accuracy. Use the evaluation report in
[`accuracy.md`](accuracy.md), which records average reward, survival steps, and
collision-free rate for repeatable greedy evaluations:

```powershell
python scripts/evaluate_model.py --episodes 10 --steps 300
```

Run this after a meaningful training change and before publishing a new
baseline. Compare rows using the same seeds and episode limit; higher reward,
longer survival, and a higher collision-free rate indicate improvement.

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
Published SARSA baseline | `published/checkpoints/sarsa_q_table.npy` |
Local/Render SARSA checkpoint | `$ARENA_STORAGE_DIR/artifacts/checkpoints/sarsa_q_table.npy` |
| Browser-started run history | `artifacts/runs/<browser-run-id>/training_history_<run>.jsonl` |
Browser-started run metadata | `artifacts/runs/<browser-run-id>/training_history_<run>.meta.json` |
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
- [Free Render Web Service](docs/render_free_web_service.md)
- [Project status](docs/timeline.md)
- [Performance history](accuracy.md)

## Thanku
Made with heart by Priyanshu ❤️
