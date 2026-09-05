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

1. Open a terminal and run the batch trainer:
   ```bash
   python scripts/train.py --agent BOTH --episodes 100 --warm-start --batch-size 64
   ```
2. The `--warm-start` flag will automatically load your human demonstrations to prefill the DQN replay buffer and initialize the SARSA Q-table.
3. The system will autonomously simulate the episodes in the background at maximum speed (no rendering overhead).
4. The script will save checkpoints to `artifacts/checkpoints` periodically.

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
