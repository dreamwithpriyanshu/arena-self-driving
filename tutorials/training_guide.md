# Training Manual

This manual explains how to use the Native PyGame simulation and CLI to train both the **DQN (Vehicle R)** and **SARSA (Vehicle S)** agents.

## 1. Human Demonstration Phase
Before autonomous training begins, you must provide a human baseline.
Since Web Browser rendering is too slow for 60FPS physics, you will record demonstrations natively using PyGame.

1. Open a terminal and run:
   ```bash
   python scripts/play_human.py R
   ```
   (Replace `R` with `S` to record for SARSA).
2. Press `ENTER` to start the PyGame window.
3. Use the arrow keys (⬆️ Accelerate, ⬇️ Brake, ⬅️ Left, ➡️ Right) to drive the car safely.
4. To safely exit and discard a bad run, press `ESC`.
5. Survive until the time limit or crash to finish the episode.
6. Aim for at least 10–20 good demonstrations per agent.

> **Pro Tip:** You can dial the difficulty up or down by passing arguments:
> `python scripts/play_human.py R --vehicles-count 30 --vehicles-density 1.5`

## 2. Autonomous Training (CLI)
Once you have recorded demonstrations, you can train the agents using the powerful, headless CLI orchestrator.

1. Open a terminal and run the batch trainer. The trainer exposes several useful options to tune learning and runtime behaviour:
   ```bash
   python scripts/train.py \
     --agent BOTH \
     --episodes 100 \
     --warm-start \
     --batch-size 64 \
     --lr 1e-3 \
     --gamma 0.99 \
     --target-update-freq 100 \
     --buffer-capacity 10000 \
     --device auto \
     --save-freq 5 \
     --history-mode per-run
   ```

   Important flags:
   - `--lr`: optimizer learning rate (affects DQN training stability).
   - `--gamma`: discount factor for future rewards.
   - `--target-update-freq`: how many DQN optimization steps between target-network updates.
   - `--buffer-capacity`: replay buffer capacity for DQN prefill and sampling.
   - `--device`: `auto` (default), `cpu`, or `cuda`.
   - `--history-mode`: `per-run` (default) creates a timestamped history file in `artifacts/` for each run; `append` writes to the legacy `artifacts/training_history.jsonl`.

2. The `--warm-start` flag will automatically load your human demonstrations to prefill the DQN replay buffer and initialize the SARSA Q-table.
3. The trainer runs headless (no rendering) at full speed for fastest training. The native PyGame rendering modes are available via `scripts/play_agent.py` for visual debugging and live training.
4. Checkpoints are saved to `artifacts/checkpoints` periodically (controlled by `--save-freq`).

### Training history files
By default the trainer writes a per-run, timestamped NDJSON file to `artifacts/` named like `training_history_YYYYMMDD_HHMMSS.jsonl`. The trainer also saves a companion metadata file `training_history_YYYYMMDD_HHMMSS.meta.json` containing the CLI arguments and run id. The Streamlit dashboard will detect recent run files and let you load them from the sidebar; the sidebar shows the run timestamp and CLI args for each run.

If you prefer the old single-file behaviour, run with `--history-mode append` and the trainer will append episode records to `artifacts/training_history.jsonl`. Even in append mode a per-run metadata file is created so runs remain identifiable.

### Greedy / deterministic runs
Use `--greedy` to run training with a greedy policy (epsilon=0). This is useful for deterministic evaluation runs or debugging when you want no exploration:

```bash
python scripts/train.py --agent R --episodes 50 --greedy --history-mode per-run
```

Note: greedy mode disables exploration by setting epsilon to 0 and disabling decay for the run.

### Example: quick training with GPU
```bash
python scripts/train.py --agent R --episodes 200 --warm-start --device cuda --lr 5e-4 --buffer-capacity 20000
```

### Stopping and resuming
- Ctrl+C during training will safely stop the loop and save checkpoints.
- Re-run the trainer in the same working directory to continue from saved checkpoints; the script attempts to load existing weights at startup.

## 3. Live AI Training (Native PyGame)
If you prefer to physically watch the agent learn and make mistakes in real-time:
1. Open a terminal and run with the `--train` and `--save` flags:
   ```bash
   python scripts/play_agent.py R --train --save
   ```
2. The agent will drive autonomously, actively exploring the environment ($\epsilon > 0$) and updating its neural networks / Q-tables after every step.
3. When the PyGame window is closed, it will automatically save the new checkpoints.

To watch BOTH agents drive in the same simulation concurrently and learn:
```bash
python scripts/play_multi_agent.py --train --save --duration 500
```

## 4. Evaluation & Analytics
To see the results of your training without any random exploration (pure exploitation):
1. Run the agent natively without the `--train` flag:
   ```bash
   python scripts/play_agent.py R
   ```

2. To dig deeper into the agent's internal learning state, visit the **Performance Analytics** page in the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```
   - **Global Comparison**: Accuracy & Survival Steps for R and S on the same charts.
   - **DQN**: Average Loss and Epsilon decay.
   - **SARSA**: Average TD Error and Epsilon decay.
   - **Raw History**: The tabular data of all recorded episodes.
