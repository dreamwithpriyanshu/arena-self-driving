# Free Render Web Service Setup

This is the dashboard-only setup for deploying Arena Self-Driving as a
standard free Render Web Service. Do not attach a persistent disk; persistent
disks are paid and are not available on the free plan.

## Create the service

1. Push the repository to GitHub.
2. In Render, choose **New > Web Service**.
3. Connect the GitHub repository and select the branch to deploy.
4. Use these exact settings:

| Dashboard field | Value |
|---|---|
| Name | `arena-self-driving` |
| Region | Choose the closest region |
| Branch | Your deployment branch, usually `main` |
| Root Directory | Leave blank |
| Runtime | `Python 3.13.7` |
| Build Command | `pip install -r requirements-render.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Instance type | `Free` |
| Health Check Path | `/` |
| Auto-Deploy | `Yes` |

Under **Environment Variables**, add:

```text
ARENA_STORAGE_DIR=/tmp/arena-data
```

The hosted requirements intentionally exclude direct PyGame installation.
`highway-env` still declares PyGame as a runtime dependency, so `.python-version`
pins Render to Python 3.13.7, which has a compatible Linux PyGame wheel.

Do not add a disk or change the mount path on the free plan.

## What works on free hosting

- Browser UI and documentation
- Headless SARSA training
- Headless SARSA evaluation
- Live training charts while the service is awake
- Published baseline checkpoint bundled in the GitHub repository

## Free-tier storage warning

`/tmp/arena-data` is temporary. Render may remove it when the service sleeps,
restarts, or is redeployed. Consequently, training performed on the free
service is not guaranteed to remain available. A new instance starts from the
tracked `published/` baseline, not from the previous free-instance checkpoint.

The free service is suitable for demonstrations and short experiments. For
durable training history and checkpoints, use a paid persistent disk or move
storage to an external database/object-storage service.

## First test

After the first deploy:

1. Open the Render URL.
2. Start one or two training episodes.
3. Confirm the live charts update.
4. Click **SARSA playback** to run headless evaluation.
5. Confirm the evaluation appears under saved runs.

Do not use **Human demonstration** on Render. It requires a local keyboard and
PyGame display. Record demonstrations locally and publish or transfer the
validated JSONL files separately.

## Publishing a free-instance improvement

Because the free filesystem is temporary, download the trained checkpoint
before the service restarts if you want to keep it. Place it locally under
`artifacts/checkpoints/`, then run:

```powershell
python scripts/publish_model.py
git add published
git commit -m "Publish improved SARSA baseline"
git push
```

The next Render deploy will start from that published baseline.

If Render still selects Python 3.14, pull the commit containing
`.python-version` and trigger a **Manual Deploy** with **Clear build cache &
deploy**.
