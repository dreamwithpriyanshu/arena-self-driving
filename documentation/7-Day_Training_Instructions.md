# 7-Day Human Training Plan — Self-Driving Car Simulation MVP

## Purpose

The coding and application development should be completed **before this 7-day period**. The seven days below are for **human-driven training and refinement of the two learning vehicles**.

The human drives inside HighwayEnv. Every driving step becomes a training example containing the current state, the human action, the reward and the next state.

The same human-generated dataset is used to train both vehicles:

- **R → DQN**: human transitions are loaded into the replay buffer and used to update the neural network.
- **S → SARSA**: human trajectories are replayed using the observed next human action for the SARSA update.

After the human session for a day, both agents can run autonomous episodes to continue learning. Keep the human dataset and autonomous data stored separately.

## Before Day 1: Finish coding in 5 build steps

Antigravity should complete the software first. Do not use the seven training days as coding days.

### Build Step 1 — Environment + state

Create the HighwayEnv setup, two controlled vehicles R and S, shared actions and the compact state representation.

### Build Step 2 — Human recorder

Create keyboard control and save every human-driven transition.

### Build Step 3 — R and S learners

Implement DQN for R and tabular SARSA for S. Both must accept the human demonstration dataset.

### Build Step 4 — Joint training + Streamlit

Run R and S together, add training/evaluation scripts and build the Streamlit interface.

### Build Step 5 — Testing + handoff

Run smoke tests, fix integration issues, verify saved checkpoints and prepare a clean GitHub-ready repository.

Only after these five build steps are working should the 7-day human-training cycle begin.

---

# Day 1 — Baseline human driving

### Goal
Create the first clean dataset and establish baseline behaviour.

### Human task
Drive one controlled vehicle at a time through normal highway traffic. Focus on safe lane keeping, acceleration, slowing and simple lane changes.

### Target
- 10–15 short episodes
- Mix of R and S control
- No deliberate risky driving
- Save every state/action/reward transition

### After the session
- Train R on the Day 1 demonstrations.
- Train S on the Day 1 demonstrations.
- Run a small autonomous evaluation for both.
- Save baseline metrics.

### Output
`day1_human.jsonl`, `day1_metrics.json`, updated R and S checkpoints.

---

# Day 2 — Following and speed control

### Goal
Teach the agents how a human reacts to vehicles ahead.

### Human task
Create situations involving:
- safe following distance
- slower vehicle ahead
- braking and recovery
- acceleration after the road clears

### Target
10–20 episodes with a mixture of traffic density and speeds.

### After the session
Append the new demonstrations to the training pool, retrain/warm-start both agents, then run the same fixed evaluation set used on Day 1.

### Output
Day 2 demonstrations + updated checkpoints + comparison against Day 1.

---

# Day 3 — Lane-change decisions

### Goal
Collect clear human examples of when a lane change is useful and when it is unsafe.

### Human task
Drive through scenarios with:
- blocked lane
- free adjacent lane
- occupied adjacent lane
- faster traffic approaching from behind
- return to a safer lane

### Target
10–20 episodes with several intentional but safe lane-change decisions.

### After the session
Retrain both agents using all demonstrations collected through Day 3.

### Output
Lane-change-rich training dataset + updated R and S models.

---

# Day 4 — Dense traffic

### Goal
Teach the models to handle more difficult traffic conditions.

### Human task
Collect demonstrations in denser traffic and avoid collisions while maintaining reasonable speed.

### Target
10–20 episodes.

### After the session
Run autonomous evaluation and specifically record:
- collision count
- survival time
- average speed
- lane changes
- action distribution

### Output
Dense-traffic dataset + Day 4 evaluation report.

---

# Day 5 — Edge cases and recovery

### Goal
Give the models difficult examples that expose weak decisions.

### Human task
Collect situations involving:
- sudden slowdown ahead
- limited lane availability
- vehicles close behind
- repeated braking
- recovering from a poor lane position

Keep scenarios safe and simulated; do not intentionally create collisions just for data.

### After the session
Retrain both agents and compare them against the Day 1 baseline using the same evaluation seeds.

### Output
Edge-case demonstrations + updated models + trend report.

---

# Day 6 — High-quality demonstrations

### Goal
Clean the dataset and collect the best examples rather than simply collecting more examples.

### Human task
Drive deliberately and consistently. Prefer decisions that are easy to explain:

`observe → choose action → remain safe → continue`

### Data work
- Remove corrupted/incomplete episodes.
- Check that actions match the allowed action set.
- Check for missing states or rewards.
- Keep human data separate from evaluation runs.

### After the session
Perform the final training pass using the curated human dataset collected through Day 6.

### Output
`human_demonstrations_final.jsonl` + final candidate checkpoints.

---

# Day 7 — Final human reference + final model comparison

### Goal
Freeze the dataset and produce the final college-project result.

### Human task
Collect a small final reference set under normal and moderately dense traffic.

### Freeze
Do not keep changing the model after the final evaluation begins.

### Final evaluation
Run fixed-seed tests for:

1. untrained/random baseline
2. trained **R — DQN**
3. trained **S — SARSA**
4. human reference episodes

### Report
Measure:
- mean episode reward
- collision rate
- survival time
- average speed
- lane changes
- action distribution
- DQN training loss
- SARSA Q-value/update statistics

### Output
- final R DQN checkpoint
- final S SARSA checkpoint
- final human dataset
- final comparison charts
- final evaluation report
- GitHub-ready release

---

## Daily training rule

Every day follows the same simple cycle:

```text
Human drives
    ↓
Record state + action + reward + next state
    ↓
Add to human training dataset
    ↓
Train R with DQN
Train S with SARSA
    ↓
Run autonomous episodes
    ↓
Measure results
    ↓
Use the next day's human driving to improve weak behaviour
```

## Important comparison rule

R and S should receive the **same human demonstrations, same state representation, same action set, same reward definition and same evaluation seeds**. The algorithm should be the main controlled difference.

Do not manufacture results. Report whatever the simulator actually produces.
