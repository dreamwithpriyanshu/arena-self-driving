# Project Timeline

## Sources and limits

This timeline is derived primarily from `git log`: commit messages, commit dates, and changed files. Existing documentation was used only to name current components. Chat-session content was unavailable, so it is not used to infer motivations or fill gaps. All listed commits are dated 2026-09-06; no more precise date, rationale, or sequence is invented here.

## 1. Foundation: environment and state

`66137a5` (`project skeleton & setup`) created the base configuration, environment package, action definitions, state builder, and environment manager. Affected files included `configs/default_env.yaml`, `src/envs/actions.py`, `src/envs/highway_factory.py`, `src/envs/state_builder.py`, and `src/simulation/env_manager.py`.

## 2. Data and human control

`d1b1095` (`Database Layer`) added episode schemas, JSONL recording, validation, loading, keyboard control, and episode management in `src/data/` and `src/human/`. `830e153` (`human layer built`) extended that layer. `963cb26` (`added human native play`) added `scripts/play_human.py`; `0d1119d` records the later WASD profile update in `configs/control_bindings.json`, `src/human/control_bindings.py`, and native scripts.

## 3. R and S learners

`eeceae0` (`Agents sarsa & dqn`) added `src/agents/base.py`, `src/agents/dqn.py`, `src/agents/sarsa.py`, and algorithm notes. The commit message identifies the learners but does not state further design rationale.

## 4. Training orchestration and Streamlit

`c13fa65` (`Training Orchestrator`) added `src/training/orchestrator.py` and connected it to the data loader. `9cbe8b9` (`Streammlit Frontend built`) added the original Streamlit pages and prior simulation/training facades. `bd498a0` added dashboard styling and `30d4b14` expanded documentation.

## 5. QA and optimization pass

`e2da947` and `3d6b9ce` are labelled `Qa completed` and `Final QA done`. Their changed files include environment setup, old simulation facades, runtime smoke scripts, README, requirements, and documentation. `2f95fb8` then addressed a dark-mode and SVG-loading issue in the earlier web UI.

## 6. Streamlit-to-native-PyGame pivot

`efb0126` (`Removed web simulations to native`) removed old Streamlit simulation/live-tracking pages and simulation facades while changing `scripts/play_agent.py`, `scripts/play_human.py`, `scripts/play_multi_agent.py`, environment setup, and README. Current Streamlit files (`app.py` and `pages/`) are data/documentation views, while native PyGame scripts run driving and playback. `b40c118` and `fc36749` record follow-up native-play fixes.

## 7. Per-run history, controls, and multi-agent robustness

`063735f` added persisted training history and updated DQN checkpoint loading. `88b440e` exposed training controls and timestamped per-run history; `6c91d93` added greedy mode and run metadata. `c43eeea` and `f513db1` fixed handling of ego-only multi-agent observations, with the latter updating tests and command/quickstart documentation. Recent commits `02dcad3`, `93a3d5f`, `0d1119d`, and `fc90d61` record native GUI controls, multi-agent fixes, WASD controls, and a speed fix. Their messages do not provide further reasoning beyond those descriptions.
