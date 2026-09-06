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

## Human demonstrations

```powershell
python scripts/play_human.py
```

The orange car is the human-controlled S vehicle. Blue cars are NPC traffic.
Use Arrow Left/Right for lane changes, Up to accelerate, and Down to slow down.
Press `S` after an episode completes to save the demonstration, or `X` to
discard it.

Saved demonstrations are JSONL files in:

```text
data/human_demonstrations/
```

They do not change the Q-table until a training run uses `--warm-start`.
Only demonstrations whose metadata is labeled `S` are used. Legacy files from
older model labels are ignored.

## Train SARSA from scratch

```powershell
python scripts/train.py --episodes 50 --seed 1000
```

This creates a new zero-initialised Q-table in memory, runs 50 autonomous
episodes, updates the table after every environment decision, and saves the
checkpoint every five episodes and again at the end.

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
