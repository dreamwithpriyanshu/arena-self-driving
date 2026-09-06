# Command Reference — Controls & Runner Commands

This file lists the primary CLI commands and flags available in the project to control the environment, change traffic and vehicle counts, run training, and operate the native PyGame viewers. Use this as a quick reference for running experiments locally.

Notes
- These examples assume you're in the repository root: `D:\RAY\Projects\arena-self-driving`.
- On Windows, activate the virtualenv (if used) before running any Python commands:

  ```powershell
  venv\Scripts\activate
  ```

- Several scripts write artifacts under `artifacts/` and `data/`. Ensure the Streamlit dashboard and trainer run in the same working directory if you want them to share those artifacts.

---

Table of contents
- Global: Streamlit & tests
- scripts/play_human.py — record human demonstrations (native)
- scripts/play_agent.py — watch a single trained agent (native)
- scripts/play_multi_agent.py — watch/train two agents simultaneously (native)
- scripts/train.py — headless training CLI (most options)
- Advanced notes: environment config and defaults

---

Global

- Start Streamlit dashboard:

  ```powershell
  streamlit run app.py
  ```

  The Streamlit app is a data command center: it shows dataset summaries, performance analytics, and allows loading per-run training history files from `artifacts/`.

- Run the test suite:

  ```powershell
  pytest tests/ -v
  ```


scripts/play_human.py

Purpose: Record human demonstrations using a native PyGame window (recommended for 60 FPS data collection).

Usage:

```powershell
python scripts/play_human.py <vehicle> [options]
```

Positional arguments:
- vehicle: `R` for DQN vehicle, `S` for SARSA vehicle

Options:
- `--vehicles-count N` : Number of NPC vehicles on the road (default: 15)
- `--duration N` : Max episode duration in steps (default: 120)
- `--vehicles-density D` : Traffic density multiplier (default: 1.0)

Examples:
- Record demonstrations for DQN (vehicle R):

  ```powershell
  python scripts/play_human.py R --vehicles-count 20 --duration 150
  ```

- Record for SARSA with denser traffic:

  ```powershell
  python scripts/play_human.py S --vehicles-density 1.5
  ```

Controls in the PyGame window:
- Arrow Up: accelerate
- Arrow Down: brake
- Arrow Left / Right: lane change
- ESC: quit and discard run
- ENTER (on instruction screen): start recording


scripts/play_agent.py

Purpose: Open a native PyGame window and watch one agent drive. Optionally enable live training (--train) and save checkpoints on exit.

Usage:

```powershell
python scripts/play_agent.py <vehicle> [options]
```

Positional arguments:
- vehicle: `R` for DQN, `S` for SARSA

Options:
- `--vehicles-count N` : Number of NPC vehicles (default: 15)
- `--duration N` : Max duration in steps (default: 120)
- `--vehicles-density D` : Traffic density multiplier (default: 1.0)
- `--train` : Enable live updates (agent will explore and call update())
- `--save` : Save checkpoints to `artifacts/checkpoints` after the run

Examples:
- Watch trained DQN agent (greedy evaluation):

  ```powershell
  python scripts/play_agent.py R
  ```

- Watch and train SARSA live, saving checkpoints at the end:

  ```powershell
  python scripts/play_agent.py S --train --save
  ```

Notes:
- When `--train` is not provided, the agent runs in evaluation (greedy) mode.
- If no checkpoint exists the agent will act randomly; use `--save` with `--train` to persist learned weights.


scripts/play_multi_agent.py

Purpose: Run a native PyGame session where BOTH DQN and SARSA drive concurrently. Useful for side-by-side comparison and live multi-agent training.

Usage:

```powershell
python scripts/play_multi_agent.py [options]
```

Options:
- `--vehicles-count N` : Number of NPC vehicles (default: 20)
- `--duration N` : Max duration in steps (default: 120)
- `--vehicles-density D` : Traffic density multiplier (default: 1.0)
- `--train` : Enable live training for both agents
- `--save` : Save checkpoints after running

Example:

```powershell
python scripts/play_multi_agent.py --vehicles-count 30 --duration 300 --train --save
```

Notes & robustness:
- The multi-agent script accepts observations either as full kinematics matrices `(V, F)` or ego-only vectors `(F,)`. Ego-only observations are automatically expanded to `(V, F)` by padding neighbour rows with zeros.
- If you prefer richer neighbour information rather than zero padding, the environment configuration may be adjusted to return the full Kinematics observation.


scripts/train.py — Headless Trainer

Purpose: The main headless orchestrator used for large-batch training. Use for fast, reproducible training runs that produce checkpoint files and per-run history files consumable by the Streamlit dashboard.

Usage:

```powershell
python scripts/train.py [options]
```

Key options (full set supported, shown with defaults):

- `--agent {R,S,BOTH}` (default: BOTH)
  - Which agent(s) to train: R = DQN, S = SARSA, BOTH = both sequentially

- `--episodes N` (default: 10)
  - Number of episodes to run

- `--warm-start` (flag)
  - Pre-fill DQN replay buffer and warm-start SARSA from human demonstrations

- `--batch-size N` (default: 64)
  - DQN replay batch size for each optimization step

- `--lr FLOAT` (default: 1e-3)
  - Learning rate for DQN optimizer (Adam)

- `--gamma FLOAT` (default: 0.99)
  - Discount factor for future rewards

- `--target-update-freq N` (default: 100)
  - Steps between copying policy_net -> target_net in DQN

- `--buffer-capacity N` (default: 10000)
  - Replay buffer size for DQN

- `--device {auto,cpu,cuda}` (default: auto)
  - Device to run PyTorch on. `auto` selects CUDA if available otherwise CPU

- `--epsilon-start FLOAT` (default: 1.0)
  - Initial exploration rate (epsilon)

- `--epsilon-decay FLOAT` (default: 0.995)
  - Multiplicative decay applied to epsilon after each step

- `--greedy` (flag)
  - Run with greedy policy (epsilon=0, decay disabled). Useful for deterministic evaluation runs

- `--vehicles-count N` (default: 15)
  - Number of NPC vehicles in the environment (influences difficulty)

- `--duration N` (default: 120)
  - Max duration (steps) per episode

- `--vehicles-density FLOAT` (default: 1.0)
  - Traffic density multiplier

- `--save-freq N` (default: 5)
  - Save model checkpoints every N episodes

- `--log-interval N` (default: 1)
  - Print metrics every N episodes

- `--history-mode {per-run,append}` (default: per-run)
  - `per-run`: create a timestamped `artifacts/training_history_<ts>.jsonl` per run and a companion `*.meta.json` with CLI args
  - `append`: append episode records to `artifacts/training_history.jsonl` (legacy mode). Even in append mode a per-run meta file is written.

- `--history-dir PATH` (default: artifacts)
  - Directory to store history files and metadata

Examples

- Quick local run (per-run history):

  ```powershell
  python scripts/train.py --agent BOTH --episodes 50 --warm-start --history-mode per-run
  ```

- GPU run for DQN only with custom buffer and learning rate:

  ```powershell
  python scripts/train.py --agent R --episodes 200 --device cuda --lr 5e-4 --buffer-capacity 20000
  ```

- Deterministic evaluation run (greedy):

  ```powershell
  python scripts/train.py --agent R --episodes 10 --greedy --history-mode per-run
  ```

Outputs
- Checkpoints: `artifacts/checkpoints/` (`dqn_policy.pt`, other agent artifacts)
- Per-run history: `artifacts/training_history_<YYYYMMDD_HHMMSS>.jsonl` (newline-delimited JSON). A companion `training_history_<ts>.meta.json` contains CLI args and run metadata.


Advanced: Environment configuration

The environment defaults live in `configs/default_env.yaml`. Key keys you can override via script `--vehicles-count` or programmatically via `config_overrides` include:

- `lanes_count`: number of lanes on the highway (default 4)
- `vehicles_count`: number of vehicles visible in the Kinematics observation (ego + neighbours, default 6)
- `vehicles_density`: traffic density multiplier
- `observation`: controls the observation type (Kinematics), `vehicles_count` seen by the agent, and `features` returned per vehicle
- `duration`: episode duration
- `render_mode`: `rgb_array` (headless) or `human` (PyGame window)

If you need to permanently change defaults for many runs, edit `configs/default_env.yaml`. For one-off runs, pass `config_overrides` via the script flags (the provided CLI flags already map to the most common overrides).


Troubleshooting & best practices
- If you get a ValueError about expected a 2-D observation, either increase the `observation.vehicles_count` in the YAML or run the native scripts with `--vehicles-count` to ensure the environment returns the full `(V, F)` Kinematics matrix.
- Use `--warm-start` to bootstrap training with human demonstrations stored under `data/human_demonstrations/`.
- Keep the Streamlit app and training scripts in the same repo working directory to ensure `artifacts/` is shared for history and checkpoints.
- Use `--history-mode per-run` to keep experiments isolated and reproducible (one file per run).

---

If you'd like, I can also:
- Add a short `tutorials/command_quickstart.md` that contains 1‑line recipes for common tasks (collect data, quick train, view dashboard), or
- Add example run metadata prints to README next to the `streamlit run app.py` line.

