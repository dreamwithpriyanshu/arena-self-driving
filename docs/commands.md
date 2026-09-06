# Command Reference

Use the native dependency set for desktop simulation:

```powershell
pip install -r requirements-native.txt
```

| Command | Purpose |
|---|---|
| `python scripts/play_human.py` | Record arrow-key driving in the PyGame GUI. |
| `python scripts/train.py --episodes 50 --warm-start --seed 1000` | Warm-start and train SARSA. |
| `python scripts/train.py --evaluation-only --resume --seed 2000` | Evaluate the saved Q-table without learning. |
| `python scripts/play_agent.py` | Watch the SARSA policy in the PyGame GUI. |
| `python -m backend.main` | Open the local training and analytics frontend at `http://127.0.0.1:8000` (binds to loopback only). |
| `pytest tests/ -v` | Run the test suite. |

The human GUI uses Arrow Left/Right/Up/Down for driving. `P` pauses, `H` shows
or hides the HUD, `+`/`-` adjusts target speed, `/` switches road pace, and
`ESC` discards or closes the current session.
