# Security, Validation, and Data Notes

## Security posture

This is a local-first, single-trusted-user educational application. It is not
currently an authenticated multi-user service. The main security boundary is
that browser input is validated and converted into server-owned process
arguments rather than executed as arbitrary shell text.

## Input and process controls

- `TrainingRequest` uses `extra="forbid"`.
- `agent_type` is restricted to `S`.
- Episodes are limited to `1..10,000`.
- Learning rate and epsilon decay are greater than zero and at most `1`.
- Gamma and epsilon values are limited to `0..1`.
- Vehicles are limited to `1..50`.
- Duration is limited to `1..5,000`.
- Seeds are limited to non-negative 32-bit values.
- `epsilon_end` cannot exceed `epsilon_start`.
- Run IDs are generated with `uuid.uuid4().hex`.
- History directories are created below the server-owned runs directory.
- Child processes use fixed executable/script paths and `shell=False`.
- Client values are passed as individual arguments, never shell fragments.
- Documentation uses an ID allowlist rather than a client filesystem path.
- NumPy checkpoints load with `allow_pickle=False`.
- Only one browser training worker is active at a time.
- Native registries prevent duplicate human or agent windows.

These controls protect the local control surface from accidental misuse and
common command/path injection mistakes. They are not a replacement for
authentication.

## Filesystem and data safety

The application persists data under the resolved storage root:

```text
data/human_demonstrations/*.jsonl
artifacts/checkpoints/sarsa_q_table.npy
artifacts/runs/<run-id>/*
published/*
accuracy.md
```

`ARENA_STORAGE_DIR` is an operator-controlled environment variable. Do not set
it to a shared or sensitive directory without checking filesystem permissions.
The application does not encrypt demonstrations, checkpoints, or metrics.

Human demonstrations can contain driving behavior, seeds, configuration values,
and timestamps. Treat them as project data and avoid publishing private
information inside metadata or paths.

Starting training without `--resume` replaces the runtime checkpoint when a new
checkpoint is saved. It does not delete demonstrations, old histories, source
code, or published files. Back up the NumPy file before intentionally starting
a fresh model.

## API exposure risks

The local server binds to `127.0.0.1`. Binding it to `0.0.0.0` exposes
endpoints that can start and terminate processes, so do not do that on an
untrusted network without adding:

- Authentication and authorization
- CSRF protection for browser state-changing requests
- Origin checking for WebSockets
- TLS
- Rate limiting
- Audit logging
- Per-user job ownership
- Resource quotas and cleanup

Hosted Render operation should be treated as a single trusted deployment. The
service does not provide accounts, roles, or tenant isolation.

## Threat-to-control table

| Risk | Current control | Remaining limitation |
|---|---|---|
| Shell injection | Fixed argv and `shell=False` | No auth if externally exposed |
| Path traversal | Server-generated run paths and doc allowlist | Operator controls storage root |
| Malicious checkpoint pickle | `allow_pickle=False` | NumPy file integrity is not signed |
| Oversized training request | Pydantic numeric bounds | A valid maximum run can still consume resources |
| Duplicate workers | In-memory job registry | Registry resets on backend restart |
| Partial live JSONL | Defensive line parsing and polling fallback | Invalid persisted records need validation |
| Sensitive demo data | Local filesystem storage | No encryption or access-control layer |
| WebSocket misuse | Opaque run IDs and server lookup | No user ownership/authentication |
| Hosted data loss | Published baseline seeding | Temporary hosted storage is not durable |

## Validation and tests

The repository has 19 automated tests:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m compileall -q backend scripts src tests
git diff --check
```

| Test area | What it verifies |
|---|---|
| `tests/test_agents.py` | SARSA action selection, updates, save/load behavior |
| `tests/test_backend.py` | FastAPI contracts, validation, run and documentation behavior |
| `tests/test_data.py` | Demonstration schemas, JSONL recording/loading, validation, summaries |
| `tests/test_envs.py` | Environment creation, actions, state conversion, dimensions |
| `tests/test_training.py` | Training/evaluation orchestration and checkpoint flow |
| `tests/test_viewer_controls.py` | Viewer controls, target speed, road mode, duration behavior |

These tests cover application behavior; they are not a penetration test and do
not make the service suitable for public exposure.

## Safe operational practices

- Keep local development bound to `127.0.0.1`.
- Use a dedicated experiment directory for destructive fresh training.
- Back up `artifacts\checkpoints\sarsa_q_table.npy` before replacing it.
- Review `accuracy.md` before publishing a model.
- Do not commit private demonstrations or unreviewed checkpoints.
- Do not treat `--seed` as a backup or model version.
- Stop only the specific process that owns the project port or run.
- Keep dependencies pinned and review changes to launcher commands.

## Shared deployment

The current service is intended for one trusted user. A shared deployment would
need authentication, job ownership, origin checks, encrypted storage, resource
limits, audit logging, and a persistent worker queue.
