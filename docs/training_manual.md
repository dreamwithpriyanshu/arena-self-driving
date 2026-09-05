# Training Manual

This manual explains how to use the Self-Driving Car MVP UI to train both the **DQN (Vehicle R)** and **SARSA (Vehicle S)** agents.

## 1. Human Demonstration Phase
Before autonomous training begins, you must provide a human baseline.

1. Navigate to the **Human Training** page.
2. Select the agent you want to record for (DQN or SARSA).
3. Click **Start Recording**.
4. Use the on-screen buttons (⬆️ Accelerate, ⬇️ Brake, ⬅️ Left, ➡️ Right, ⏸️ IDLE) to drive the car safely.
5. If you crash, click **Discard Episode** (bad data hurts training).
6. If you survive a decent amount of time, click **Save Episode**.
7. Aim for at least 10–20 good demonstrations per agent.

## 2. Warm-Starting
Once you have recorded demonstrations:
1. Navigate to the **Train & Compare** page.
2. Click **Warm-Start (Human Demos)**.
3. This pre-fills the DQN replay buffer and initializes the SARSA Q-table based on your safe driving data, giving the agents a massive head start.

## 3. Autonomous Training
1. Stay on the **Train & Compare** page.
2. Set the `Episodes per batch` slider (e.g., 5 or 10).
3. Click **Train DQN (R)** or **Train SARSA (S)**.
4. The system will autonomously simulate the episodes.
5. Watch the **Live Performance Curves** (Reward and Survival Steps) update incrementally after every episode.
6. Periodically click **Save Checkpoints** to ensure you don't lose progress.

## 4. Live Tracking
If you want to physically watch the agent drive:
1. Navigate to the **Live Tracking** page.
2. Select your agent.
3. Adjust the **Playback Speed**.
4. Click **Run Episode** (which includes exploration/learning) or **Evaluate** (greedy, pure exploitation).
5. Watch the top-down SVG rendering and live telemetry (Speed, Lane, Reward) update in real time.

## 5. Analytics
To dig deeper into the agent's internal learning state, visit the **Performance Analytics** page to view:
- **DQN**: Average Loss and Epsilon decay.
- **SARSA**: Average TD Error and Epsilon decay.
- **Raw History**: The tabular data of all recorded episodes.
