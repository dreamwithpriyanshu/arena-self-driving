# Commands and SARSA Workflow

Run commands from the repository root with the virtual environment active.

## Install and start the workbench

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m backend.main
```

Open `http://127.0.0.1:8000`. The browser can start local human recording,
headless training, and local agent playback.

If port 8000 is already in use, reuse the existing server or stop only the
specific process holding that port before restarting. Do not start duplicate
servers and expect them to share jobs.

## Human demonstrations (required for warm-start)

```powershell
python scripts/play_human.py
```

The orange car is the human-controlled S vehicle. Blue cars are NPC traffic.
Use Arrow Left/Right for lane changes, Up to accelerate, and Down to slow down.
When the episode completes, press `S` to save it, `X` to discard it, or `Esc`
to close without saving. Saving is required: training cannot warm-start from
an episode that was not saved.

Saved demonstrations are JSONL files in:

```text
data/human_demonstrations/
```

They do not change the Q-table until a training run uses `--warm-start`.
To verify that a recording was saved, check that a new `.jsonl` file exists in
that directory or check the demonstration count in the browser under Drive and
play.
Only demonstrations whose metadata is labeled `S` are used. Legacy files from
older model labels are ignored.

## Train SARSA from scratch (new runtime checkpoint)

```powershell
python scripts/train.py --episodes 50 --seed 1000
```

This creates a new zero-initialised Q-table in memory, runs 50 autonomous
episodes, updates the table after every environment decision, and saves the
checkpoint every five episodes and again at the end. It replaces the one
runtime checkpoint at `artifacts/checkpoints/sarsa_q_table.npy` when complete,
but it does not delete demonstrations, previous run history, or published
files.

For a longer run:

```powershell
python scripts/train.py --episodes 500 --duration 300 --seed 1000
```

## Continue training an existing model

```powershell
python scripts/train.py --episodes 100 --resume --seed 2000
```

`--resume` loads the existing `sarsa_q_table.npy` before training. Without it,
the command starts a fresh table even if a checkpoint already exists.

`--seed` does not mean “make a new model.” It seeds Python, NumPy, and each
episode (`seed`, `seed + 1`, and so on) so a run can be repeated. A different
seed changes the experience order but does not delete data. Use `--resume` with
the seed to continue the current table, or omit `--resume` to intentionally
train a fresh table and replace the runtime checkpoint.

### What is preserved when starting fresh?

When `--resume` is omitted, `train.py` starts a zero-initialised Q-table. It
does not remove the project data. The old runtime table at
`artifacts/checkpoints/sarsa_q_table.npy` is replaced when checkpoints are
saved, but these remain untouched:

- `data/human_demonstrations/*.jsonl`
- Earlier `artifacts/training_history_*.jsonl` and metadata
- `published/`
- Code and configuration files

The old runtime Q-table is not automatically backed up. Copy it first if you
may need to restore it:

```powershell
Copy-Item artifacts\checkpoints\sarsa_q_table.npy artifacts\checkpoints\sarsa_q_table.backup.npy
python scripts/train.py --episodes 100 --seed 2000
```

For a completely separate experiment, use another storage root instead of
overwriting the current runtime checkpoint:

```powershell
$env:ARENA_STORAGE_DIR = "D:\arena-experiment-2"
python scripts/train.py --episodes 100 --seed 2000
Remove-Item Env:ARENA_STORAGE_DIR
```

Include human demonstrations before autonomous learning:

```powershell
python scripts/train.py --episodes 100 --resume --warm-start --seed 2000
```

Warm-start applies SARSA updates to validated recorded transitions. It is a
starting bias, not a permanent copy of the human policy.

## Useful training commands

```powershell
# More traffic and a longer episode
python scripts/train.py --episodes 200 --resume --vehicles-count 25 --vehicles-density 1.5 --duration 500

# Change learning behaviour
python scripts/train.py --episodes 200 --resume --lr 0.05 --gamma 0.95 `
  --epsilon-start 0.8 --epsilon-end 0.05 --epsilon-decay 0.995

# Deterministic greedy command-line evaluation; does not learn
python scripts/train.py --evaluation-only --resume --episodes 10 --duration 300 --seed 3000

# Do not explore during a training-style run (normally use evaluation-only)
python scripts/train.py --episodes 10 --resume --greedy --seed 3000
```

PowerShell accepts a backtick for line continuation. The commands above can
also be written on one line.

## Evaluate the saved model

```powershell
python scripts/evaluate_model.py --episodes 10 --steps 300 --seed 1000
```

Evaluation loads the checkpoint, switches the agent to greedy mode, runs the
requested episodes without updates, and appends average reward, survival steps,
and collision-free rate to `accuracy.md`.

It does not improve or overwrite the model. Use the same seed, episode count,
and step limit when comparing two checkpoints.

## Inspect local playback

```powershell
# Greedy native viewer
python scripts/play_agent.py

# View while continuing to train interactively
python scripts/play_agent.py --train --save
```

In the agent viewer, `+/-` changes target speed, `/` changes road pace, `P` or
Space pauses, `H` toggles the HUD, and `Q/E` lower or raise epsilon only when
`--train` is active. Normal playback is greedy.

The viewer places NPC traffic both ahead of and behind the ego vehicle so lane
changes can be evaluated in both directions. The `+/-` control allows
`8–40 m/s`, matching HighwayEnv's vehicle maximum of `40 m/s`. The configured
reward range tops out at `30 m/s`, so speeds above 30 remain physically valid
but do not receive additional high-speed reward.

## Backend and hosted operation

The browser calls FastAPI; it does not train in JavaScript:

```text
Browser -> FastAPI -> scripts/train.py -> EnvManager -> HighwayEnv
                                      -> SARSA Q-table
                                      -> JSONL history/checkpoint
```

The backend validates all training values, creates an opaque run directory,
starts a fixed argument list with `shell=False`, and streams JSONL metrics over
WebSockets. If the socket is unavailable, the browser polls saved metrics.
Only one browser training worker can run at once.

For a hosted headless process:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Human PyGame recording is local-only. Hosted agent playback runs one headless
evaluation. Hosted files may be temporary; publishing remains an explicit
checkpoint-copy and Git workflow.

## Publish a checkpoint to GitHub

Publishing is deliberately explicit; commits do not automatically publish a
model.

```powershell
python scripts/evaluate_model.py --episodes 10 --steps 300 --seed 1000
python scripts/publish_model.py
git add published accuracy.md
git commit -m "Publish improved SARSA baseline"
git push
```

`publish_model.py` copies the current runtime checkpoint and the newest
command-line history into `published/`. Review the evaluation first. A code
commit can be made without publishing a model, and evaluation alone does not
publish anything.

## Storage locations

| Data | Location | Purpose |
|---|---|---|
| Human demonstrations | `data/human_demonstrations/*.jsonl` | Optional warm-start transitions |
| Runtime model | `artifacts/checkpoints/sarsa_q_table.npy` | Model being trained or evaluated |
| Run history | `artifacts/training_history_*.jsonl` | Per-episode metrics |
| Run metadata | `artifacts/training_history_*.meta.json` | Arguments and run mode |
| Published model | `published/checkpoints/sarsa_q_table.npy` | GitHub-shared baseline |
| Performance report | `accuracy.md` | Evaluation timeline |

Set `ARENA_STORAGE_DIR` to move writable data, for example:

```powershell
$env:ARENA_STORAGE_DIR = "D:\arena-data"
python scripts/train.py --episodes 100 --resume
```

The published baseline is copied into a new storage directory automatically.
Existing runtime files are not overwritten.

## Script behavior summary

- `backend.main` serves the browser and starts validated child processes.
- `play_human.py` records and saves keyboard demonstrations only when `S` is
  pressed after completion.
- `train.py` trains SARSA, optionally loads a checkpoint with `--resume`, and
  optionally consumes saved demonstrations with `--warm-start`.
- `play_agent.py` displays the saved agent; its `--train --save` mode can also
  update the checkpoint interactively.
- `evaluate_model.py` loads the checkpoint in greedy mode and only appends
  aggregate results to `accuracy.md`.
- `publish_model.py` copies the runtime checkpoint and latest history into
  `published/`; Git commands are still manual.
- `pytest` and `compileall` validate code and do not train the model.

## Documentation and UI

The browser Documentation section loads the allowlisted Markdown pages returned
by `GET /documentation`. The current UI uses a warm neutral palette, muted
teal primary actions, responsive cards, normal scrollable documentation, and
short hover/focus transitions. It intentionally avoids neon colors and heavy
animation.

Available documentation includes:

```text
project, architecture, algorithm, commands, backend, security, timeline, ui
```

## What happens during one training episode?

1. The environment resets with the selected traffic, timing, and random seed.
2. SARSA converts the observation into a discrete state containing lane, speed,
   front gap, front speed difference, adjacent-lane safety, and rear gap.
3. The policy selects one action: lane left, idle, lane right, faster, or slower.
4. HighwayEnv advances the traffic and returns the next state and reward.
5. SARSA selects the next action and applies the on-policy update:

   ```text
   Q(s,a) += learning_rate * (reward + gamma * Q(next_state,next_action) - Q(s,a))
   ```

6. Epsilon decays after the update. The episode ends on collision or when its
   decision limit is reached.
7. Reward, steps, TD error, epsilon, and average Q values are written to JSONL.
8. The Q-table is checkpointed according to `--save-freq` and at shutdown.

## Factors that affect learning

| Factor | Effect |
|---|---|
| `--episodes` | More episodes provide more updates and state coverage. |
| `--duration` | More decisions per episode allow longer manoeuvres and more updates. |
| `--vehicles-count` | More NPCs create more traffic situations and collisions. |
| `--vehicles-density` | Higher density packs traffic closer together and increases difficulty. |
| `--lr` | Learning rate; high values learn faster but can be unstable, low values are slower. |
| `--gamma` | Future-reward importance; high values favour long-term survival and progress. |
| `--epsilon-start` | Initial random exploration. |
| `--epsilon-end` | Minimum exploration retained during training. |
| `--epsilon-decay` | How quickly exploration falls; smaller values become greedy sooner. |
| `--collision-reward` | Makes crashes more or less costly. |
| `--high-speed-reward` | Encourages safe progress at the configured speed range. |
| `--right-lane-reward` | Encourages the preferred right lane. |
| `--lane-change-reward` | Penalises or rewards lane changes; the default discourages weaving. |
| `--simulation-frequency` | Physics updates per second. |
| `--policy-frequency` | How often actions are selected; it cannot exceed simulation frequency. |
| `--seed` | Makes experiments more repeatable. |
| `--warm-start` | Uses human transitions to give the initial Q-table a demonstrated bias. |
| State bins in `configs/default_env.yaml` | Decide which situations share one Q-value. Coarse bins generalise more; fine bins distinguish more situations. |

Training is not conventional accuracy classification. Improvement is measured
by higher comparable reward, longer survival, and a higher collision-free rate.

## Validation and cleanup

```powershell
python -m pytest
python -m compileall -q backend scripts src tests
```

The model is not automatically published by either command.
