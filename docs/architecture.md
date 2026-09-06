# Architecture

The submission has two deliberately separate runtime surfaces.

```text
Arrow-key PyGame GUI -> human recorder -> JSONL demonstrations
                                      -> SARSA trainer -> Q-table and history
Streamlit dashboard ----------------------------------> reads evidence only
```

## Modules

- `src/envs/`: HighwayEnv configuration, actions, and state discretisation.
- `src/simulation/`: environment reset, step, metrics, and resource lifecycle.
- `src/human/`: arrow-key capture and recorded episode lifecycle.
- `src/data/`: JSONL schemas, validation, loading, and dashboard helpers.
- `src/agents/sarsa.py`: the sole tabular on-policy learner.
- `src/training/`: headless SARSA warm start, training, evaluation, and
  checkpointing.

The native GUI is intentionally separate from Streamlit. A browser-hosted
dashboard cannot reliably provide a PyGame window on Community Cloud, while a
read-only evidence dashboard remains fast and deployable.

## Training contract

The environment exposes five discrete HighwayEnv meta-actions: left, idle,
right, faster, and slower. SARSA receives a compact discrete state, chooses an
epsilon-greedy action, and updates with the selected next action. Human episode
trajectories preserve their subsequent action during warm start.
