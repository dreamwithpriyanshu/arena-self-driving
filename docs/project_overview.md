# Project Overview — Self-Driving Car Simulation MVP

## What this is

A simulation-based comparison of two reinforcement learning algorithms
applied to self-driving on a highway:

- **Vehicle R** — trained with **DQN** (Deep Q-Network), an off-policy
  neural-network Q-learner using experience replay.
- **Vehicle S** — trained with **SARSA** (State-Action-Reward-State-Action),
  an on-policy tabular learner.

Both agents learn from the **same human demonstrations** collected inside
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv), a lightweight
Gymnasium-compatible highway driving simulator.  The algorithm is the only
controlled variable — environment, actions, state representation, reward
function, and evaluation seeds are identical.

## Why it matters

This is a college-level project designed to:

1. Show how the same human-driving data can bootstrap two fundamentally
   different RL algorithms.
2. Make the comparison **watchable** — a reviewer can open the app and see
   both cars driving live, not just read log files.
3. Produce reproducible, transparent results with no manufactured numbers.

## Technology stack

| Component | Technology |
|-----------|-----------|
| Simulation engine | HighwayEnv + Gymnasium |
| RL — DQN | PyTorch |
| RL — SARSA | Pure NumPy (tabular) |
| UI | Streamlit |
| Charts | Plotly / Altair |
| Data format | JSONL (human demos) + PyTorch checkpoints |

## Build phases

The software is built in **5 steps** (see `mvp.md`) before a 7-day
human-training cycle begins:

1. Environment, actions, and state representation
2. Human recorder and keyboard control
3. R (DQN) and S (SARSA) agent implementations
4. Joint training, Streamlit pages, live tracking, analytics
5. QA, documentation, and handoff

## Repository layout

See `docs/architecture.md` for the full layer diagram and module map.
