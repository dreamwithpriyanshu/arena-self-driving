# SARSA Algorithm Notes

## 1. Purpose and scope

Arena Self-Driving uses one learning algorithm: tabular SARSA
(State-Action-Reward-State-Action). The `S` label in the UI, demonstration
metadata, and run metadata always means this policy.

SARSA is used here because its table is small, inspectable, and easy to connect
to a simulated driving environment. It is not a neural network, does not use
gradient descent, and does not claim to model real-world vehicle dynamics.

The learning system has four main responsibilities:

1. Convert a HighwayEnv observation into a discrete state index.
2. Select an action with an epsilon-greedy policy.
3. Update one Q-table entry after each environment step.
4. Save and reload the table so training can continue later.

## 2. Reinforcement-learning terminology

The implementation uses the following terms:

| Term | Meaning in this project |
|---|---|
| Environment | HighwayEnv highway simulation |
| Agent | `SARSAAgent` in `src/agents/sarsa.py` |
| State | Discrete description of lane, speed, gaps, and lane safety |
| Action | One of five high-level driving actions |
| Reward | Scalar feedback from HighwayEnv |
| Episode | One reset-to-terminal/truncated driving run |
| Transition | `(state, action, reward, next_state, done)` record |
| Q-value | Estimated future return for a state/action pair |
| Q-table | NumPy matrix containing all Q-values |

The agent does not predict a steering angle or raw acceleration. It chooses
high-level meta-actions, and HighwayEnv applies the corresponding vehicle
behavior.

## 3. Environment observation

The default environment uses HighwayEnv's normalised Kinematics observation:

```text
[x, y, vx, vy, cos_heading, sin_heading]
```

The configured observation contains six vehicle rows:

- Row 0: the ego vehicle
- Rows 1–5: the nearest observed vehicles

The default configuration uses relative positions and normalised features. The
raw observation is flattened and retained in transition records for inspection,
but the Q-table uses the compact discrete state index.

The environment configuration is in `configs/default_env.yaml`:

```yaml
observation:
  type: "Kinematics"
  vehicles_count: 6
  absolute: false
  normalize: true
```

The state builder also accepts one-dimensional observations by expanding them
into a six-row matrix and zero-padding neighbour rows. Normal operation uses
the full Kinematics matrix.

## 4. Discrete state representation

The state builder extracts seven categorical features:

| Feature | Categories | Default source |
|---|---:|---|
| Ego lane | 4 | True lane index from `EnvManager` |
| Ego speed | 3 | `speed_bins: [0.33, 0.66]` |
| Front gap | 3 | `front_gap_bins: [0.2, 0.5]` |
| Front speed difference | 3 | `front_speed_diff_bins: [-0.1, 0.1]` |
| Left lane safety | 2 | Free or blocked |
| Right lane safety | 2 | Free or blocked |
| Rear gap | 2 | `rear_gap_bins: [0.3]` |

The total number of states is:

```text
4 × 3 × 3 × 3 × 2 × 2 × 2 = 864
```

The Q-table therefore has this shape:

```text
(864 states, 5 actions)
```

The agent validates this shape when loading a checkpoint. A checkpoint created
with incompatible state bins, lane count, or action count is rejected instead
of being used silently.

### 4.1 Speed category

The ego vehicle's normalised longitudinal speed is digitised with
`[0.33, 0.66]`:

```text
0 = slow
1 = normal
2 = fast
```

### 4.2 Front vehicle and front gap

The state builder examines neighbour rows with positive relative `x` and
selects the closest one. Its relative longitudinal gap is digitised with:

```text
[0.2, 0.5]
```

The resulting categories are:

```text
0 = close
1 = medium
2 = far
```

If no vehicle is ahead, the state uses the highest category, representing a
clear/far front path.

### 4.3 Front speed difference

The nearest front vehicle's longitudinal speed feature is digitised with:

```text
[-0.1, 0.1]
```

This produces:

```text
0 = slower
1 = approximately similar
2 = faster
```

If no front vehicle exists, the highest category is used because there is no
front obstacle constraining the ego vehicle.

### 4.4 Adjacent-lane safety

The state builder checks each adjacent lane independently. A lane is marked
unsafe when an observed vehicle is both:

- In the approximate adjacent-lane band
- Within the configured close longitudinal gap

The encoded values are:

```text
0 = blocked / not free
1 = free
```

The left and right indicators are separate, so the policy can distinguish
between a safe left change, a safe right change, and a situation where both
changes are unsafe.

### 4.5 Rear gap

The nearest vehicle with negative relative `x` is selected. Its absolute rear
gap is digitised with `[0.3]`:

```text
0 = close rear vehicle
1 = safe or no rear vehicle
```

This feature is important for avoiding a lane change into traffic approaching
from behind.

## 5. State-index encoding

The seven categorical values are combined into one integer with mixed-radix
encoding. In conceptual form:

```text
state = encode(
    lane,
    speed,
    front_gap,
    front_speed_difference,
    left_free,
    right_free,
    rear_gap
)
```

The implementation multiplies each feature by the product of the sizes of the
features that follow it. This gives every valid combination one stable integer
between `0` and `863`.

The true lane index comes from the environment manager rather than the
normalised relative observation. This avoids treating the ego vehicle's
relative `y` value (which is normally near zero) as its absolute lane.

## 6. Action space

The action enum is defined in `src/envs/actions.py` and maps directly to
HighwayEnv's `DiscreteMetaAction` indices:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `LANE_LEFT` | Request a lane change to the left |
| 1 | `IDLE` | Maintain the current tactical action |
| 2 | `LANE_RIGHT` | Request a lane change to the right |
| 3 | `FASTER` | Request higher speed |
| 4 | `SLOWER` | Request lower speed |

The agent does not issue multiple actions simultaneously. Human input can
produce a held action list, but the recorder applies the first resolved action
for each decision interval.

## 7. Q-table

`SARSAAgent` creates a zero-initialised NumPy table:

```python
q_table = np.zeros((864, 5), dtype=np.float32)
```

Each row represents one discrete driving situation. Each column represents one
available action.

For example:

```text
Q[350, LANE_RIGHT]
```

is the current estimated return for taking a right lane change in state 350.

The table is stored at:

```text
artifacts/checkpoints/sarsa_q_table.npy
```

The table is not a neural-network model and has no learned weights beyond these
state/action values.

## 8. Epsilon-greedy action selection

During training, `SARSAAgent.act()` uses epsilon-greedy selection:

1. If evaluation mode is enabled, select a greedy action.
2. Otherwise, draw a random value.
3. With probability epsilon, choose a random action.
4. Otherwise, choose an action with the highest Q-value.

When multiple actions have exactly the same maximum Q-value, the
implementation chooses randomly among those tied actions. This prevents a
fixed action-order bias when the table is initially all zeros.

Default exploration settings:

```text
epsilon_start = 1.00
epsilon_end   = 0.05
epsilon_decay = 0.995
```

After each non-evaluation update:

```text
epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

The result is high exploration early in training and increasingly greedy
behavior later, while retaining a small exploration floor.

## 9. SARSA update rule

SARSA is an on-policy temporal-difference method. For a transition
`(s, a, r, s')`, the policy selects the next action `a'`, then updates:

```text
Q(s, a) ← Q(s, a)
          + learning_rate *
            (r + gamma * Q(s', a') - Q(s, a))
```

The temporal-difference error is:

```text
TD error = r + gamma * Q(s', a') - Q(s, a)
```

The implementation uses:

```text
learning_rate = 0.10
gamma         = 0.99
```

These can be changed with `train.py` arguments:

```powershell
python scripts/train.py --episodes 200 --lr 0.05 --gamma 0.95
```

### 9.1 Non-terminal transition

For a normal transition:

1. The current action is executed.
2. The environment returns reward and next state.
3. The policy chooses the next action from the next state.
4. The next action's Q-value is included in the target.
5. The current table entry is updated.

### 9.2 Terminal transition

For a collision, truncation, or final step, there is no future action value:

```text
Q(next_state, next_action) = 0
```

The target becomes:

```text
target = reward
```

This prevents value from leaking beyond an ended episode.

## 10. One training episode

`TrainingOrchestrator.train_episode()` performs this sequence:

1. Set the agent to training mode.
2. Reset the environment with the requested seed.
3. Select the first action using the current epsilon-greedy policy.
4. For each step:
   - Save the previous result.
   - Apply the current action through `EnvManager.step()`.
   - Mark the final allowed step as truncated when needed.
   - Build a `Transition`.
   - Select the next action unless the episode ended.
   - Apply one SARSA update.
   - Accumulate TD error and average Q metrics.
   - Invoke the optional live callback.
   - Continue with the next action.
5. Return total reward, steps, termination state, duration, epsilon, and
   aggregate learning metrics.

The command-line trainer writes each episode record to JSONL immediately and
flushes the file, which allows the browser to display live progress.

## 11. Reward signals

The default reward configuration is:

```yaml
collision_reward: -2.0
right_lane_reward: 0.1
high_speed_reward: 0.4
lane_change_reward: -0.05
reward_speed_range: [20, 30]
```

The intended behavior is:

- Strongly discourage collisions
- Reward safe progress at a useful speed
- Slightly prefer the right lane
- Discourage unnecessary lane weaving

The reward is not an accuracy score. It is feedback used to update Q-values.
Two runs should only be compared when their environment settings, episode
limits, and evaluation seeds are comparable.

## 12. Human demonstrations and warm-start

Human demonstrations are stored as validated JSONL episodes. Each transition
contains:

```text
state
action
reward
next_state
terminated
truncated
discrete_state
next_discrete_state
lane
speed
```

When `--warm-start` is enabled:

1. The loader scans the demonstration directory.
2. Episode files are validated.
3. Invalid files can be skipped.
4. Only metadata labeled vehicle `S` is selected.
5. The recorded transitions are replayed through SARSA updates.
6. For each non-final demonstration transition, the following recorded action
   is supplied as `next_action`.
7. Autonomous training then continues from the warm-started table.

Warm-start is a learning bias, not behavior cloning. The table remains
trainable, and later autonomous experience can reinforce or reduce the effect
of demonstrated actions.

## 13. Training versus evaluation

Training mode:

```powershell
python scripts/train.py --episodes 100 --resume
```

- Sets evaluation mode to false
- Allows exploration
- Applies Q-table updates
- Decays epsilon
- Saves checkpoints

Evaluation mode:

```powershell
python scripts/train.py --evaluation-only --resume --episodes 10 --duration 300
```

- Loads the checkpoint
- Sets evaluation mode to true
- Uses greedy action selection
- Does not update Q-values
- Does not improve or overwrite the model

The dedicated evaluation script additionally aggregates reward, survival steps,
and collision-free rate into `accuracy.md`.

## 14. Checkpoint lifecycle

Save:

```python
agent.save(checkpoint_directory)
```

This writes `sarsa_q_table.npy`.

Load:

```python
agent.load(checkpoint_directory)
```

Loading checks that the table shape equals `(864, 5)`. A missing file raises
`FileNotFoundError`; an incompatible shape raises `ValueError`.

`--resume` loads the table before the first training episode. Without
`--resume`, the agent begins with zeros and the newly saved table replaces the
runtime checkpoint. Demonstrations and previous run history are separate files
and are not deleted by this operation.

## 15. What the metrics mean

| Metric | Meaning |
|---|---|
| `total_reward` | Sum of rewards in one episode |
| `steps` | Number of environment decisions |
| `epsilon` | Exploration probability after updates |
| `avg_td_error` | Mean signed TD error for the episode |
| `avg_q` | Mean Q-table value sampled after updates |
| `terminated` | Episode ended naturally, usually collision |
| `truncated` | Episode reached an external/time limit |

TD error can be positive or negative. A lower absolute TD error often means
the table is changing less, but it is not by itself proof of better driving.
Use reward, survival, and collision-free rate together.

## 16. Reproducibility

When `--seed N` is supplied:

- Python's random generator is seeded
- NumPy's random generator is seeded
- Episode `1` uses seed `N`
- Episode `2` uses seed `N + 1`
- Later episodes continue that sequence

The seed does not create a separate checkpoint and does not delete data. For a
fair comparison, use the same seed, episode count, duration, traffic settings,
and evaluation procedure.

## 17. Known design limits

- The table only represents the configured discrete state features.
- States outside the configured bins are grouped into boundary categories.
- The observation contains only the configured nearest vehicles.
- A changed lane count or bin configuration changes the required Q-table shape.
- Training quality depends on reward design, traffic density, seeds, and
  checkpoint state.
- Simulation performance does not establish real-world driving safety.

## 18. Practical commands

Train from the current checkpoint:

```powershell
python scripts/train.py --episodes 100 --resume --seed 2000
```

Train with demonstrations:

```powershell
python scripts/train.py --episodes 100 --resume --warm-start --seed 2000
```

Start fresh in a separate storage directory:

```powershell
$env:ARENA_STORAGE_DIR = "D:\arena-experiment-2"
python scripts/train.py --episodes 100 --seed 2000
Remove-Item Env:ARENA_STORAGE_DIR
```

Evaluate:

```powershell
python scripts/evaluate_model.py --episodes 10 --steps 300 --seed 1000
```

Run visual playback:

```powershell
python scripts/play_agent.py
```

Run tests:

```powershell
python -m pytest
```
