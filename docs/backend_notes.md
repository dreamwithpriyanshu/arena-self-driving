# FastAPI Control Surface

Run the local server from the project root:

```powershell
python -m backend.main
```

It listens on `127.0.0.1:8000` only. Open `http://127.0.0.1:8000` in the same
machine's browser. Do not expose this server to a network without adding
authentication, TLS, and an authorization design for process launch requests.

For Render, use the `render.yaml` Blueprint. Render is headless: the human
PyGame endpoint returns `501`, while the agent endpoint starts a one-episode
headless evaluation and saves its metrics as a normal run.

Render training does not push files to GitHub. The persistent disk belongs to
the Render service; publishing a checkpoint to GitHub is an explicit download,
`publish_model.py`, commit, and push workflow.

## API contract

| Endpoint | Purpose |
|---|---|
| `POST /sessions/human/start` | Open the native PyGame human demonstration recorder. |
| `POST /sessions/agent/start` | Open the native PyGame SARSA playback viewer. |
| `GET /demonstrations/summary` | Return validated demonstration totals. |
| `POST /training/start` | Start one headless SARSA run. |
| `GET /training/status/{run_id}` | Return the in-memory job status and process ID. |
| `POST /training/stop/{run_id}` | Terminate the active training process. |
| `GET /runs` | List saved run metadata from `artifacts/`. |
| `GET /runs/{run_id}/metrics` | Load saved JSONL metrics for one run. |
| `WS /ws/training/{run_id}` | Stream newly written episode metrics and status. |
| `GET /documentation` | List frontend documentation pages. |

`POST /training/start` accepts only agent type `S`. The request model rejects
unknown fields and validates ranges: episodes 1-10,000, learning rate `(0, 1]`,
gamma and epsilon values `[0, 1]`, epsilon decay `(0, 1]`, vehicles 1-50,
duration 1-5,000, and a non-negative 32-bit seed. `epsilon_end` cannot exceed
`epsilon_start`. Browser training resumes from the saved checkpoint by default;
the request may disable this with `resume: false`.

The backend builds a fixed `list[str]` for `scripts/train.py` and runs it with
`shell=False`. It does the same for the two native PyGame launchers. Client
input is never used as a shell command or file path.

## Run lifetime

Only one browser-started training process may be active at a time. Its status
is `running`, `completed`, `failed`, or `stopped`. The registry is intentionally
in memory: restarting the FastAPI server clears active-job status, but saved
history remains available through `/runs`.
