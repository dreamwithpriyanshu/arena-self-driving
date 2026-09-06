# Algorithm Notes: Tabular SARSA

SARSA is the only learner in this project. It stores expected action values in
a NumPy Q-table indexed by a discretised driving state and one of five actions.

For each transition, it uses the on-policy update:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q(s_next, a_next) - Q(s, a))
```

`a_next` is the next action selected by the current policy. During recorded
human warm start, the next demonstrated action is used when available.

This choice is appropriate for the submission because it is inspectable, fast
on a CPU, and does not require neural-network or GPU dependencies.
