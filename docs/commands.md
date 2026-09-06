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

  Arrow keys are the default drive profile; press `C` on the setup screen for
  WASD. `P` pauses without creating a recorded transition. See
  `configs/control_bindings.json` to remap either profile.

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
  python scripts/train.py --agent BOTH --episodes 100 --warm-start --seed 1000
  ```

Run metadata example (created alongside each per-run history file):

```json
{
  "run_id": "20260906_145000",
  "mode": "training",
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
