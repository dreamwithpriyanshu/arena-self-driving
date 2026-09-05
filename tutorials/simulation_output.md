# Understanding Simulation Output

The Self-Driving Car MVP uses a custom, lightweight SVG renderer in Streamlit to display the highway environment in real time. This avoids heavy 3D rendering overhead.

## The Live Tracking View

![Live Tracking](screenshots/3_live_tracking.png)

### Visual Elements
- **Cyan Car (Vehicle R)**: Represents the DQN agent.
- **Orange Car (Vehicle S)**: Represents the SARSA agent.
- **Gray Cars**: Background NPC traffic (obstacles).
- **Lanes**: The highway consists of 4 lanes. The ego vehicle must learn to navigate around slower NPC traffic by changing lanes without colliding.

### Telemetry Readouts
- **Speed (m/s)**: Current velocity of the ego vehicle.
- **Lane**: Current lane index (0 = top, 3 = bottom).
- **Reward**: Cumulative reward for the episode. Positive for moving forward, negative for collisions.
- **Loss / TD Error**: The current internal learning metric of the network/table. Drops as the model converges.

## Interpreting Agent Behavior
- **Exploration (Early Training)**: The agent will frequently change lanes randomly and likely crash. This is normal ($\epsilon$ is high).
- **Exploitation (Late Training)**: The agent will stick to a fast lane and only change lanes when an NPC is blocking its path.
- **DQN vs SARSA**: You will notice DQN (R) is capable of smoother multi-lane planning, whereas SARSA (S) might exhibit slightly more rigid, bucketed state responses due to its discrete table.

Check the **Performance Analytics** page to see the exact convergence curves!

![Performance Analytics](screenshots/4_performance_analytics.png)
