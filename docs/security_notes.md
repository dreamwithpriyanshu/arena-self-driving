# Security and Data Notes

- Demonstration and artifact writes are constrained to project `data/` and
  `artifacts/` directories.
- Episode filenames are sanitised before writing.
- JSONL demonstrations are validated before training.
- SARSA checkpoints use NumPy arrays with `allow_pickle=False` when loading.
- The Streamlit dashboard reads evidence; it does not execute simulator or
  training actions from browser input.
