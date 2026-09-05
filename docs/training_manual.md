# Training Manual

This manual explains how to use the Native PyGame simulation and Streamlit Dashboard to train both the **DQN (Vehicle R)** and **SARSA (Vehicle S)** agents.

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

## 2. Warm-Starting
Once you have recorded demonstrations:
1. Start the Data Dashboard: `streamlit run app.py`
2. Navigate to the **Model Training** page.
3. Click **Warm-Start (Human Demos)**.
4. This pre-fills the DQN replay buffer and initializes the SARSA Q-table based on your safe driving data, giving the agents a massive head start.

## 3. Autonomous Training
1. Stay on the **Model Training** page in Streamlit.
2. Set the `Episodes per batch` slider (e.g., 5 or 10).
3. Click **Train DQN (R)** or **Train SARSA (S)**.
4. The system will autonomously simulate the episodes in the background at maximum speed (no rendering overhead).
5. Watch the **Live Performance Curves** (Reward and Survival Steps) update incrementally after every episode.
6. Periodically click **Save Checkpoints** to ensure you don't lose progress.

## 4. Live Tracking / Agent Evaluation
If you want to physically watch the agent drive:
1. Open a terminal and run:
   ```bash
   python scripts/play_agent.py R
   ```
2. The agent will drive autonomously using the weights saved in `artifacts/checkpoints`.
3. To watch BOTH agents drive in the same simulation concurrently:
   ```bash
   python scripts/play_multi_agent.py
   ```

## 5. Analytics
To dig deeper into the agent's internal learning state, visit the **Performance Analytics** page in the Streamlit Dashboard to view:
- **Global Comparison**: Accuracy & Survival Steps for R and S on the same charts.
- **DQN**: Average Loss and Epsilon decay.
- **SARSA**: Average TD Error and Epsilon decay.
- **Raw History**: The tabular data of all recorded episodes.
