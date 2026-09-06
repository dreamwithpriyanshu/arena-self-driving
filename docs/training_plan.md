# Seven-Day SARSA Training Plan

## Aim

This plan is for improving the saved SARSA policy over seven focused days.
It cannot guarantee 100% accuracy or collision-free driving. A perfect score
on a small fixed test set can still fail on new traffic layouts, so the main
goal is reliable behavior across unseen seeds and traffic conditions.

## Day 1 — Establish a baseline

Back up the current checkpoint, run a baseline training session, and evaluate
it with fixed seeds. Record average reward, survival steps, and collision-free
rate in `accuracy.md`.

```powershell
Copy-Item artifacts\checkpoints\sarsa_q_table.npy `
  artifacts\checkpoints\sarsa_q_table.day1.backup.npy
python scripts/train.py --episodes 100 --resume --seed 1000
python scripts/evaluate_model.py --episodes 20 --steps 300 --seed 1000
```

## Day 2 — Collect better demonstrations

Record 20–50 careful human episodes. Save smooth, collision-free drives and
discard poor demonstrations. Then warm-start the existing table:

```powershell
python scripts/play_human.py
python scripts/train.py --episodes 200 --resume --warm-start --seed 2000
```

Demonstrations are stored as JSONL under `data/human_demonstrations/`.

## Day 3 — Increase state coverage

Train with different seeds, traffic counts, densities, and longer episodes.
The purpose is to expose the table to more lane, speed, gap, and rear-traffic
states without changing the state representation.

```powershell
python scripts/train.py --episodes 300 --resume --vehicles-count 20 `
  --vehicles-density 1.2 --duration 400 --seed 3000
```

## Day 4 — Compare SARSA settings

Run separate experiments with learning rates around `0.05–0.10`, gamma around
`0.95–0.99`, and exploration ending around `0.02–0.05`. Keep the configuration
with the best consistent collision-free rate and survival, not only the highest
single-run reward.

Use a separate storage directory when comparing fresh tables:

```powershell
$env:ARENA_STORAGE_DIR = "D:\arena-experiment-day4"
python scripts/train.py --episodes 300 --seed 4000
python scripts/evaluate_model.py --episodes 20 --steps 300 --seed 4000
Remove-Item Env:ARENA_STORAGE_DIR
```

## Day 5 — Reduce risky behavior

Use moderate target speeds, keep the lane-change penalty active, and retain a
strong collision penalty. Evaluate greedily after training. Avoid selecting a
model only because it drives faster; stable survival is more important.

```powershell
python scripts/train.py --episodes 300 --resume --duration 400 --seed 5000
python scripts/evaluate_model.py --episodes 30 --steps 400 --seed 5000
```

## Day 6 — Stress test

Evaluate on conditions that were not used for the main training run:

- Different random seeds
- More vehicles and higher density
- Different starting lanes
- Longer episodes
- Different target-speed settings

Reject models that perform well only on familiar seeds.

## Day 7 — Final evaluation and publishing

Run a larger greedy evaluation, compare it with the Day 1 baseline, and publish
only a checkpoint that improves consistently:

```powershell
python scripts/evaluate_model.py --episodes 100 --steps 300 --seed 7000
python scripts/publish_model.py
git add published accuracy.md
git commit -m "Publish improved SARSA baseline"
git push
```

## Practical success criteria

Use these measures instead of claiming absolute accuracy:

- Collision-free rate close to 100% across unseen seeds
- Longer average survival
- No major performance collapse in denser traffic
- Stable real-time playback
- Higher comparable reward than the baseline

Evaluation is meaningful only when episode count, seed, duration, and
environment settings are recorded and kept comparable.
