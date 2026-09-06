# SARSA Algorithm Notes

SARSA is the only learning algorithm in this project. The `S` label in the
frontend and recorded metadata always means the tabular SARSA policy; there is
no second agent type or model comparison mode.

## State and actions

HighwayEnv supplies a normalised Kinematics observation. The state builder
uses the ego vehicle and nearby traffic to form a discrete state from lane,
speed, front gap, front speed difference, adjacent-lane safety, and rear gap.
With the default four lanes and configured bins, the Q-table has 864 states.

SARSA chooses from five HighwayEnv meta-actions:

| ID | Action |
|---:|---|
| 0 | lane left |
| 1 | idle |
| 2 | lane right |
| 3 | faster |
| 4 | slower |

## Update rule

After a transition `(s, a, reward, s_next)`, SARSA selects `a_next` using its
current epsilon-greedy policy and applies:

```text
Q(s, a) = Q(s, a) + learning_rate * (reward + gamma * Q(s_next, a_next) - Q(s, a))
```

This is on-policy learning: the update uses the action the current policy
would actually take next. During warm start, validated human demonstrations
provide the recorded next action when it exists.

## Training and evaluation

Training starts with `epsilon_start` exploration and decays epsilon after
updates toward `epsilon_end`. The browser displays epsilon and average TD error
alongside reward and survival steps. Evaluation-only command-line runs are
greedy and do not update the table.

The Q-table is a NumPy file at `artifacts/checkpoints/sarsa_q_table.npy`.
Training overwrites that checkpoint with the latest saved policy, so use the
saved JSONL run history to compare experiments.
