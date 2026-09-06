# Command Reference (short) — available run scripts

This doc provides a compact reference for the main runtime scripts. For the full command reference see `tutorials/commands.md`.

Commands

- Start Streamlit dashboard:

  ```bash
  streamlit run app.py
  ```

- Human recording (native):

  ```bash
  python scripts/play_human.py R --vehicles-count 20
  ```

- Agent viewer (native):

  ```bash
  python scripts/play_agent.py R --train --save
  ```

- Multi-agent (native):

  ```bash
  python scripts/play_multi_agent.py --vehicles-count 30 --train --save
  ```

- Headless training (recommended):

  ```bash
  python scripts/train.py --agent BOTH --episodes 100 --warm-start --history-mode per-run
  ```

Run metadata example (created alongside each per-run history file):

```json
{
  "run_id": "20260906_145000",
  "history_mode": "per-run",
  "history_path": "artifacts/training_history_20260906_145000.jsonl",
  "args": {
    "agent": "BOTH",
    "episodes": 100,
    "warm_start": true,
    "batch_size": 64,
    "lr": 0.001,
    "device": "auto"
  },
  "created_at": 1694068200.0
}
```

This file is also intended to appear in the Streamlit Documentation page under the "Documentation" tab.
