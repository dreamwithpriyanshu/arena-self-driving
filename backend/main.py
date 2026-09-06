"""Local-only API and static frontend for controlling SARSA training."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.loader import get_dataset_summary
from src.storage import artifacts_dir, checkpoints_dir, demonstrations_dir, ensure_storage_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = artifacts_dir()
RUNS_DIR = ARTIFACTS_DIR / "runs"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCUMENTS = {
    "architecture": "architecture.md",
    "algorithm": "algorithm_notes.md",
    "commands": "commands.md",
    "security": "security_notes.md",
    "backend": "backend_notes.md",
    "timeline": "timeline.md",
    "ui": "ui_design_system.md",
}


class TrainingRequest(BaseModel):
    """Only validated values from this model are passed to the trainer."""

    model_config = ConfigDict(extra="forbid")

    agent_type: Literal["S"] = "S"
    episodes: int = Field(default=50, ge=1, le=10_000)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0)
    vehicles_count: int = Field(default=15, ge=1, le=50)
    duration: int = Field(default=300, ge=1, le=5_000)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    warm_start: bool = False
    resume: bool = True

    @model_validator(mode="after")
    def validate_schedule(self) -> "TrainingRequest":
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start")
        return self


@dataclass
class TrainingJob:
    run_id: str
    pid: int
    status: str
    created_at: float
    history_dir: str
    stopped_at: float | None = None


class JobRegistry:
    """Thread-safe, in-memory ownership and status registry for local jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def active_job(self) -> TrainingJob | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running":
                    process = self._processes.get(job.run_id)
                    if process and process.poll() is None:
                        return job
                    job.status = "failed" if process and process.returncode else "completed"
                    if job.stopped_at is not None:
                        job.status = "stopped"
            return None

    def create(self, run_id: str, process: subprocess.Popen[bytes], history_dir: Path) -> TrainingJob:
        job = TrainingJob(run_id, process.pid, "running", time.time(), str(history_dir))
        with self._lock:
            self._jobs[run_id] = job
            self._processes[run_id] = process
        return job

    def get(self, run_id: str) -> TrainingJob | None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            process = self._processes.get(run_id)
            if job.status == "running" and process and process.poll() is not None:
                job.status = "failed" if process.returncode else "completed"
                if job.stopped_at is not None:
                    job.status = "stopped"
            return job

    def stop(self, run_id: str) -> TrainingJob | None:
        with self._lock:
            job = self._jobs.get(run_id)
            process = self._processes.get(run_id)
            if job is None:
                return None
            if job.status == "running" and process and process.poll() is None:
                process.terminate()
                job.status = "stopped"
                job.stopped_at = time.time()
            return job


registry = JobRegistry()
app = FastAPI(title="Arena SARSA Training Control", docs_url=None, redoc_url=None)
ensure_storage_dirs()


class NativeSessionRegistry:
    """Tracks the desktop PyGame windows launched from the local frontend."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def start(self, mode: Literal["human", "agent"]) -> dict[str, object]:
        if os.environ.get("RENDER") == "true" and mode == "human":
            raise HTTPException(
                status_code=501,
                detail="Human keyboard demonstrations require a local PyGame window.",
            )
        if os.environ.get("RENDER") == "true":
            run_id = uuid.uuid4().hex
            history_dir = RUNS_DIR / f"evaluation-{run_id}"
            history_dir.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
                "--episodes", "1", "--evaluation-only", "--resume",
                "--history-dir", str(history_dir),
                "--checkpoint-dir", str(checkpoints_dir()),
                "--data-dir", str(demonstrations_dir()),
            ]
            try:
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, shell=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                history_dir.rmdir()
                raise HTTPException(status_code=500, detail="Unable to start hosted evaluation.") from exc
            job = registry.create(run_id, process, history_dir)
            return {"mode": mode, "hosted": True, **job_payload(job)}
        with self._lock:
            existing = self._processes.get(mode)
            if existing and existing.poll() is None:
                raise HTTPException(status_code=409, detail=f"The {mode} window is already open.")
            script_name = "play_human.py" if mode == "human" else "play_agent.py"
            try:
                process = subprocess.Popen(
                    [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
                    cwd=PROJECT_ROOT,
                    shell=False,
                )
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Unable to open the {mode} window.") from exc
            self._processes[mode] = process
            return {"mode": mode, "pid": process.pid, "status": "running"}


native_sessions = NativeSessionRegistry()


def job_payload(job: TrainingJob) -> dict[str, object]:
    return asdict(job)


def training_command(request: TrainingRequest, history_dir: Path) -> list[str]:
    """Build the complete trusted argv list; no client string becomes executable code."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--episodes", str(request.episodes),
        "--lr", str(request.learning_rate),
        "--gamma", str(request.gamma),
        "--epsilon-start", str(request.epsilon_start),
        "--epsilon-end", str(request.epsilon_end),
        "--epsilon-decay", str(request.epsilon_decay),
        "--vehicles-count", str(request.vehicles_count),
        "--duration", str(request.duration),
        "--history-dir", str(history_dir),
        "--checkpoint-dir", str(checkpoints_dir()),
        "--data-dir", str(demonstrations_dir()),
    ]
    if request.resume:
        command.append("--resume")
    if request.seed is not None:
        command.extend(["--seed", str(request.seed)])
    if request.warm_start:
        command.append("--warm-start")
    return command


def history_file_for(job: TrainingJob) -> Path | None:
    paths = sorted(Path(job.history_dir).glob("training_history_*.jsonl"))
    return paths[0] if paths else None


def read_metrics(history_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        with history_path.open(encoding="utf-8") as history:
            for line in history:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError:
                    continue  # A trainer may be flushing the final line.
    except OSError:
        pass
    return records


@app.post("/training/start", status_code=201)
def start_training(request: TrainingRequest) -> dict[str, object]:
    if registry.active_job() is not None:
        raise HTTPException(status_code=409, detail="A training run is already active.")

    run_id = uuid.uuid4().hex
    history_dir = RUNS_DIR / run_id
    history_dir.mkdir(parents=True, exist_ok=False)
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            training_command(request, history_dir),
            cwd=PROJECT_ROOT,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except OSError as exc:
        history_dir.rmdir()
        raise HTTPException(status_code=500, detail="Unable to start the training process.") from exc
    return job_payload(registry.create(run_id, process, history_dir))


@app.post("/sessions/human/start", status_code=201)
def start_human_session() -> dict[str, object]:
    """Open the existing native PyGame demonstration recorder."""
    return native_sessions.start("human")


@app.post("/sessions/agent/start", status_code=201)
def start_agent_session() -> dict[str, object]:
    """Open the existing native PyGame SARSA playback window."""
    return native_sessions.start("agent")


@app.get("/demonstrations/summary")
def demonstration_summary() -> dict[str, object]:
    """Expose validated human-driving evidence without allowing file paths."""
    return get_dataset_summary()


@app.get("/training/status/{run_id}")
def training_status(run_id: str) -> dict[str, object]:
    job = registry.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown run ID.")
    return job_payload(job)


@app.post("/training/stop/{run_id}")
def stop_training(run_id: str) -> dict[str, object]:
    job = registry.stop(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown run ID.")
    return job_payload(job)


@app.get("/runs")
def list_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for metadata_path in ARTIFACTS_DIR.rglob("training_history_*.meta.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                metadata["artifact_run_id"] = metadata_path.parent.name
                metadata["history_path"] = str(metadata_path.with_name(metadata_path.name.replace(".meta.json", ".jsonl")).relative_to(PROJECT_ROOT))
                runs.append(metadata)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return sorted(runs, key=lambda run: float(run.get("created_at", 0)), reverse=True)


@app.get("/runs/{artifact_run_id}/metrics")
def run_metrics(artifact_run_id: str) -> list[dict[str, object]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", artifact_run_id):
        raise HTTPException(status_code=404, detail="Unknown run ID.")
    for metadata_path in ARTIFACTS_DIR.rglob("training_history_*.meta.json"):
        if metadata_path.parent.name == artifact_run_id:
            return read_metrics(metadata_path.with_name(metadata_path.name.replace(".meta.json", ".jsonl")))
    raise HTTPException(status_code=404, detail="Unknown run ID.")


@app.get("/documentation")
def list_documentation() -> list[dict[str, str]]:
    return [{"id": doc_id, "filename": filename} for doc_id, filename in DOCUMENTS.items()]


@app.get("/documentation/{document_id}")
def read_documentation(document_id: str) -> dict[str, str]:
    filename = DOCUMENTS.get(document_id)
    if filename is None:
        raise HTTPException(status_code=404, detail="Unknown documentation page.")
    try:
        return {"id": document_id, "filename": filename, "content": (DOCS_DIR / filename).read_text(encoding="utf-8")}
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to read documentation.") from exc


@app.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str) -> None:
    job = registry.get(run_id)
    if job is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    sent_records = 0
    last_status: str | None = None
    try:
        while True:
            job = registry.get(run_id)
            if job is None:
                return
            history_path = history_file_for(job)
            if history_path:
                records = read_metrics(history_path)
                for record in records[sent_records:]:
                    await websocket.send_json({"type": "metric", "data": record})
                sent_records = len(records)
            if job.status != last_status:
                await websocket.send_json({"type": "status", "data": job_payload(job)})
                last_status = job.status
            if job.status != "running":
                return
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
