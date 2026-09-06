# Commands and Workflow

## Install

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements-native.txt
```

## Recommended browser workflow

```powershell
python -m backend.main
```

Open `http://127.0.0.1:8000`, then:

1. Choose **Open human drive** to collect and save arrow-key demonstrations.
2. Enable **Warm-start from saved demonstrations** if desired.
3. Configure the SARSA form and start a run.
4. Read reward, steps, epsilon, and TD-error charts as episodes complete.
5. Choose **Open agent play** to view the latest saved checkpoint.

## Direct local tools

| Command | Purpose |
|---|---|
| `python scripts/play_human.py` | Directly open the native human demonstration recorder. |
| `python scripts/play_agent.py` | Directly open the native SARSA playback viewer. |
| `python scripts/train.py --episodes 50 --warm-start --seed 1000` | Train SARSA from demonstrations. |
| `python scripts/train.py --evaluation-only --resume --seed 2000` | Greedily evaluate the saved Q-table without learning. |
| `pytest tests -q` | Run automated checks. |

The human window uses Arrow Left/Right/Up/Down to drive. `P` pauses, `H`
toggles the HUD, `+` and `-` adjust target speed, `/` switches road pace, and
`ESC` discards or closes the session.
