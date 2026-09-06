# Quickstart Recipes

Short one-line recipes for common tasks.

- Collect human demonstrations for DQN (Vehicle R):

  ```powershell
  python scripts/play_human.py R --vehicles-count 20 --duration 150
  ```

- Quick headless training run (timestamped per-run history):

  ```powershell
  python scripts/train.py --agent BOTH --episodes 50 --warm-start --seed 1000
  ```

- Evaluate both saved checkpoints without updating them on matching seeds:

  ```powershell
  python scripts/train.py --agent BOTH --episodes 10 --evaluation-only --resume --seed 2000
  ```

- Train DQN on GPU with larger buffer:

  ```powershell
  python scripts/train.py --agent R --episodes 200 --device cuda --lr 5e-4 --buffer-capacity 20000
  ```

- Watch both agents live and enable training:

  ```powershell
  python scripts/play_multi_agent.py --vehicles-count 30 --train --save
  ```

- View analytics in Streamlit:

  ```powershell
  streamlit run app.py
  ```

These commands assume you are in the repository root and have activated the virtual environment. Use `--help` on any script to see full flag lists, e.g. `python scripts/train.py --help`.
