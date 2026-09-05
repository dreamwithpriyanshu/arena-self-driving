# Understanding Simulation Output

The Self-Driving Car MVP uses a high-performance **Native PyGame** renderer to display the highway environment at a smooth 60FPS.

## The PyGame View

### Visual Elements
- **Green Car (Ego)**: Represents your controlled vehicle (Human, DQN, or SARSA).
- **Blue/Gray Cars**: Background NPC traffic (obstacles).
- **Lanes**: The highway consists of 4 lanes. The ego vehicle must learn to navigate around slower NPC traffic by changing lanes without colliding.

### Live Telemetry (Terminal)
While watching the PyGame window, check your terminal for real-time telemetry updates:
- **Steps**: Number of actions taken so far.
- **Reward**: Cumulative reward for the episode. Positive for moving forward, negative for collisions.
- **Agent Action**: The discrete integer (0-4) chosen by the agent.

## Interpreting Agent Behavior
- **Exploration (Early Training)**: If you run with the `--train` flag early on, the agent will frequently change lanes randomly and likely crash. This is normal ($\epsilon$ is high).
- **Exploitation (Late Training)**: When running normal evaluation, the agent will stick to a fast lane and only change lanes when an NPC is blocking its path.
- **DQN vs SARSA**: You will notice DQN (R) is capable of smoother multi-lane planning, whereas SARSA (S) might exhibit slightly more rigid, bucketed state responses due to its discrete table representation.

Check the **Performance Analytics** page in the Streamlit Dashboard (`streamlit run app.py`) to see the exact convergence curves!
