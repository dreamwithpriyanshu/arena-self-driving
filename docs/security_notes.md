# Security and Data Notes

- Demonstration and artifact writes are constrained to project `data/` and
  `artifacts/` directories.
- Episode filenames are sanitised before writing.
- JSONL demonstrations are validated before training.
- SARSA checkpoints use NumPy arrays with `allow_pickle=False` when loading.
- The FastAPI control surface accepts only strict, bounded SARSA
  hyperparameters and launches the trainer with a fixed subprocess argument
  list (`shell=False`). It binds to `127.0.0.1` by default.
