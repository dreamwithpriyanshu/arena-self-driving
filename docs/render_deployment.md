# Render Deployment Guide

This service is a Python FastAPI web service with a browser UI. It is not a
desktop application on Render: Render has no display or keyboard, so native
PyGame windows are unavailable there.

## Recommended setup: Blueprint

1. Push the repository to GitHub, including `render.yaml`.
2. In Render, choose **New > Blueprint** and select the repository and branch.
3. Review the generated `arena-self-driving` web service.
4. Choose a paid instance type that supports a persistent disk.
5. Apply the Blueprint and wait for the first deploy.

The Blueprint sets these values:

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Python 3.13.7 (from `.python-version`) |
| Build command | `pip install -r requirements-render.txt` |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/` |
| Persistent disk mount | `/var/data` |
| Storage environment variable | `ARENA_STORAGE_DIR=/var/data` |

Do not change the start command to `python -m backend.main` on Render unless
the `HOST` and `PORT` environment variables are configured. The explicit
Uvicorn command is the safest hosted setting.

## Manual Web Service settings

If you do not use the Blueprint, create **New > Web Service** with:

- **Repository:** this GitHub repository
- **Branch:** the branch containing this code
- **Root Directory:** blank
- **Runtime:** Python 3.13.7
- **Build Command:** `pip install -r requirements-render.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Auto-deploy:** enabled if you want GitHub pushes to deploy code changes
- **Environment variable:** `ARENA_STORAGE_DIR` = `/var/data`
- **Disk:** mount at `/var/data`; 1 GB is enough for the current tabular model,
  demonstrations, and histories

The disk is persistent across deploys and restarts, but belongs only to this
service instance. Do not scale horizontally with this file-backed design.

## First deployment checklist

After deployment, open the Render URL and verify:

1. The home page loads.
2. **Documentation** opens.
3. **Saved runs** loads without an error.
4. **Start training** with 1-2 episodes works.
5. The run reaches `completed` and appears in saved history.
6. **SARSA playback** starts a headless evaluation and adds a result.
7. Restart the service and confirm the checkpoint and saved runs remain.

The first training run resumes from the published SARSA checkpoint when one is
present. Browser training resumes from the existing checkpoint by default.
Therefore, later runs improve the current model instead of resetting it.

## Render versus GitHub

An automatic deploy pulls code from GitHub; it does not push trained files back
to GitHub. Training writes to:

```text
/var/data/artifacts/checkpoints/sarsa_q_table.npy
/var/data/artifacts/runs/
/var/data/data/human_demonstrations/
```

To release a Render-trained model as the new GitHub baseline, download the
checkpoint and latest history, place them in the corresponding local
`artifacts/` paths, then run:

```powershell
python scripts/publish_model.py
git add published
git commit -m "Publish improved SARSA baseline"
git push
```

## Headless behavior

- **Human demonstration:** returns a clear error on Render; record these
  keyboard episodes locally and transfer validated JSONL files if needed.
- **SARSA playback:** runs a one-episode, no-display evaluation and streams its
  reward, survival steps, and status to the browser charts.
- **Training:** is real SARSA training. It updates the NumPy checkpoint and
  JSONL history on the persistent disk.

## Troubleshooting

- **Build fails:** confirm the repository root contains both requirements files.
- **Service does not start:** use the exact Uvicorn start command above.
- **Runs disappear:** verify the disk is mounted at `/var/data` and
  `ARENA_STORAGE_DIR` has exactly that value.
- **Native window error:** expected on Render; use local PyGame for keyboard
  recording and hosted headless evaluation for playback.
- **Out of disk space:** increase disk size or remove old histories from the
  persistent disk; never remove the current checkpoint.
