# Algorithm Notes — Self-Driving Car Simulation MVP

This document explains the two reinforcement learning algorithms used in this project—**DQN (Vehicle R)** and **SARSA (Vehicle S)**—in plain language. 

The goal of both algorithms is the same: to learn a "policy" (a set of rules) that tells the vehicle which action (e.g., speed up, slow down, change lanes) to take in any given situation to maximize its long-term reward.

---

## 1. Deep Q-Network (DQN) — Vehicle "R"

**Concept:** 
DQN is a modern, neural-network-based approach. It looks at the continuous, raw sensor data of the world (exact positions and speeds of surrounding vehicles) and passes it through a deep neural network to predict the "Q-value" (expected future reward) for each possible action.

**How it learns:**
- **Experience Replay**: As DQN drives (or when it loads human demonstrations), it stores every transition (state, action, reward, next state) into a large "memory buffer".
- **Batch Learning**: Instead of learning from just the most recent moment, it constantly pulls random batches of past memories and trains on them. This breaks the correlation between consecutive frames and makes training much more stable.
- **Target Network**: It uses a secondary, slowly-updating neural network to estimate future rewards. This prevents the learning target from shifting too wildly while the main network updates.

**Why it's used here:**
DQN represents complex, continuous state spaces very well. It is highly capable but requires more compute (PyTorch) and can be sensitive to tuning parameters (learning rate, target network frequency).

---

## 2. Tabular SARSA — Vehicle "S"

**Concept:**
SARSA (State-Action-Reward-State-Action) is a classic, foundational reinforcement learning algorithm. Instead of a neural network, it uses a giant lookup table (a "Q-table"). 
Because a table cannot hold infinite combinations of continuous speeds and positions, the world is first "discretised". We chop the world into a fixed number of bins (e.g., 864 possible distinct situations based on nearest-neighbor positions and lane). 

**How it learns:**
- **On-Policy Tabular Updates**: SARSA does not use a replay buffer. It learns "live" as it drives. When it takes an action and sees the result, it immediately updates the specific cell in its lookup table corresponding to that exact `[State, Action]` combination.
- **Warm Starting**: To speed up training, we can pass human demonstrations through the table before autonomous training begins. The agent updates its table as if it were driving the human trajectories itself.

**Why it's used here:**
SARSA is extremely fast, transparent, and computationally cheap (implemented in pure Numpy). Because we discretised the state space heavily, it serves as a robust, predictable baseline to compare against the more complex neural network of DQN.

---

## Comparison Summary

| Feature | DQN (Vehicle R) | SARSA (Vehicle S) |
|---------|-----------------|-------------------|
| **State Representation** | Continuous vector (36 float dimensions) | Discrete index (864 integer bins) |
| **Brain** | Deep Neural Network (PyTorch) | Lookup Table (Numpy) |
| **Memory** | Experience Replay Buffer | None (updates instantly) |
| **Learning Style** | Off-policy (learns from old memories) | On-policy (learns from current actions) |
| **Human Data Usage** | Pre-fills the replay buffer | "Warm starts" by pre-updating the table |
