# End-to-End Training Guide (CLI & Native PyGame)

This walkthrough explains how to execute the entire 7-day training workflow using the new high-performance CLI workflow.

## 1. Record Human Demonstrations
Before the autonomous agents can learn efficiently, they need a safe baseline.

1. Open your terminal.
2. Run the human play script for **Agent: R (DQN)**:
   ```bash
   python scripts/play_human.py R
   ```
3. The Native PyGame window will open. Press `ENTER` and use your arrow keys to control the car safely.
4. Survive without crashing until the time limit to save the episode. If you crash, press `ESC` to discard the bad data.
5. Repeat until you have ~10-20 saved episodes.
6. Repeat the process for **Agent: S (SARSA)**: `python scripts/play_human.py S`

## 2. Headless Autonomous Training (Fastest)
The most efficient way to train the agents is via the headless CLI orchestrator.

1. In your terminal, run the batch trainer:
   ```bash
   python scripts/train.py --agent BOTH --episodes 100 --warm-start --batch-size 64
   ```
2. The `--warm-start` flag will automatically load your human demonstrations to prefill the DQN replay buffer and initialize the SARSA Q-table.
3. The script will rapidly simulate 100 episodes without rendering overhead, logging metrics and saving checkpoints periodically.

## 3. Live AI Training (Visual)
If you want to actually *watch* the agents learn in real-time, you can enable training during live evaluation!

1. Run the multi-agent visualizer with the `--train` flag:
   ```bash
   python scripts/play_multi_agent.py --train --save --duration 500
   ```
2. The PyGame window will open, and both DQN and SARSA will drive simultaneously.
3. Because `--train` is active, they will actively explore (epsilon > 0) and update their networks/tables after every step!
4. When you exit, the `--save` flag ensures their newly learned weights are saved.

## 4. Evaluate Performance & Analytics
To see the result of your training without any random exploration (pure exploitation):

1. Run the agent natively:
   ```bash
   python scripts/play_agent.py R
   ```
   (Notice the absence of `--train`, meaning it will drive using its fully optimized, greedy policy).

2. **Analytics Dashboard**: 
   Open the Streamlit app to view the performance metrics, loss curves, and side-by-side agent accuracy:
   ```bash
   streamlit run app.py
   ```
   Navigate to the **Performance Analytics** page to dive deep into the math!
