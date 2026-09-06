# SARSA Performance History

This project is reinforcement learning, not classification. It has no ground
truth label and therefore no conventional accuracy percentage. Model quality is
tracked with repeatable greedy evaluations:

- **Average reward:** mean episode return.
- **Average survival steps:** mean decisions before termination or truncation.
- **Collision-free rate:** episodes that finish without a collision termination.

Run this after training and before publishing a new baseline:

```powershell
python scripts/evaluate_model.py --episodes 10 --steps 300
```

The command appends one dated row below. Use the same episode count, step
limit, and seed when comparing model versions.

| Date (local) | Episodes | Average reward | Average survival steps | Collision-free rate |
|---|---:|---:|---:|---:|

| 2026-09-06 22:12:59 | 3 | 32.661 | 37.7 | 66.7% |

| 2026-09-06 22:17:51 | 10 | 51.484 | 60.2 | 10.0% |

| 2026-09-06 22:23:04 | 2 | 51.009 | 60.0 | 100.0% |

| 2026-09-06 22:37:22 | 2 | 26.626 | 30.0 | 100.0% |

| 2026-09-06 23:10:56 | 10 | 84.877 | 101.0 | 30.0% |
