# Understanding Native Simulation Output

The interactive simulation is a native PyGame window backed by HighwayEnv.
Streamlit does not render or control this loop.

## Human driving

`python scripts/play_human.py R` opens an instruction screen first. Press
ENTER, then use the arrow keys to control the selected vehicle. The recorder
shows the highway and traffic directly in the native window and saves the
episode when it ends. Press ESC to discard an unsafe or incomplete run.

![Native PyGame human driving](screenshots/native_human_driving.png)

This image should show the actual HighwayEnv window while a human controls the
vehicle.

## Agent playback

`python scripts/play_agent.py R` and `python scripts/play_agent.py S` open an
instruction screen that identifies the algorithm and whether the run is greedy
evaluation or exploratory training. The agent then acts every environment step
until the duration, collision, or window close ends the episode.

`python scripts/play_multi_agent.py` runs DQN and SARSA together using two
controlled vehicles. This is the visual comparison mode; the Performance
Analytics page is the historical comparison mode.

The multi-agent window includes a live telemetry HUD with step progress,
current action, cumulative reward, lane, speed, crash/timeout state, and
training status for R and S. Its controls are:

- `SPACE` — pause or resume the episode
- `H` — hide or show the telemetry HUD
- `S` — save both checkpoints immediately
- `ESC` — stop the session

![Native PyGame agent playback](screenshots/native_agent_playback.png)

Optional side-by-side reference:

![Native PyGame multi-agent playback](screenshots/native_multi_agent.png)

## Reading the result

- A collision ends the current episode and is reported in the terminal.
- A duration limit ends the episode without a collision.
- `--train` means the agent continues updating while it drives.
- `--save` writes the updated checkpoint after the native session.
- If a checkpoint is missing, the viewer reports it and the agent may act
  randomly; this is expected for a new checkout.

The dashboard reads the resulting training history with:

```powershell
streamlit run app.py
```
