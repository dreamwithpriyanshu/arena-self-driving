# Training Guide

The project uses native PyGame for interactive driving and a headless CLI for
repeatable training. Streamlit only reads the resulting datasets and history.

## 1. Record human demonstrations

Run one vehicle at a time from the repository root:

```powershell
python scripts/play_human.py R
python scripts/play_human.py S
```

The instruction window waits for ENTER. In the highway window:

- Arrow Up accelerates.
- Arrow Down brakes.
- Arrow Left and Right change lanes.
- ESC discards the active episode.

When the episode ends normally, the recorder saves a validated JSONL file in
`data/human_demonstrations/`. Adjust traffic with the flags documented in
[`tutorials/commands.md`](commands.md), for example:

```powershell
python scripts/play_human.py R --vehicles-count 30 --vehicles-density 1.5
```

## 2. Train both agents headlessly

Warm-start from the shared demonstrations and create an isolated history file:

```powershell
python scripts/train.py --agent BOTH --episodes 100 --warm-start --history-mode per-run
```

The trainer writes checkpoints to `artifacts/checkpoints/`, a timestamped
`artifacts/training_history_<run>.jsonl`, and matching `.meta.json` metadata.
Use `--device cuda` when PyTorch GPU support is available. Use `--greedy` for
a deterministic policy run.

## 3. Watch native agent playback

Watch one trained agent:

```powershell
python scripts/play_agent.py R
python scripts/play_agent.py S
```

Use `--train --save` to explore and update the selected agent while watching:

```powershell
python scripts/play_agent.py R --train --save
```

For a side-by-side DQN/SARSA session:

```powershell
python scripts/play_multi_agent.py --duration 300
```

The native window is the only live simulation view. ESC or closing the window
ends playback; `--save` persists updated checkpoints.

Reference capture from the native viewer:

![Native PyGame agent playback](screenshots/native_agent_playback.png)

## 4. Inspect evidence in Streamlit

Start the dashboard after a recording or training run:

```powershell
streamlit run app.py
```

Use the four pages as follows:

- **Overview**: confirms where evidence is stored and repeats the native commands.
- **Human Demonstrations**: counts episodes/transitions and shows metadata.
- **Performance Analytics**: loads a selected training-history file and plots
  reward, survival, loss, TD error, and exploration where those fields exist.
- **Documentation**: renders the project notes.

The dashboard never starts HighwayEnv and never fabricates missing results.
