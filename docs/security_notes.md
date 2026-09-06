# Security Notes — Self-Driving Car Simulation MVP

This document covers the security boundaries and input validation rules implemented in **Build Step 2** (Human Recorder) and applied throughout the project.

Because this project does not run a web server or handle user authentication,
the threat model focuses on **local robustness** and safe handling of
user-generated JSONL files produced by the native recorder.

---

## 1. File I/O Safety

All episode files (human demonstrations and autonomous logs) are saved locally. A malicious or malformed episode ID must not be allowed to escape the intended directories or overwrite arbitrary files on the system.

### Protections in `TransitionRecorder`
- **Path Sanitisation**: Any user-provided `episode_id` is passed through `_sanitise_filename()`, which strips all characters except `[a-zA-Z0-9_\-]`.
- **Directory Traversal Prevention**: Before opening any file, `_is_safe_path()` resolves the absolute path and verifies it sits strictly inside the project's `data/` or `artifacts/` directories. If `../../` tricks are attempted, the recorder raises a `ValueError`.
- **Frequent Flushes (Crash Resilience)**: Each transition is immediately `flush()`'d to disk. While not strictly atomic against mid-write power failures, it prevents data loss from buffering if the simulation crashes or is killed mid-episode. A crash mid-write results in a truncated JSON line, which the `validator.py` will catch and flag, keeping the system safe from corrupted inputs.
- **Max Episode Length**: The `EpisodeManager` enforces a `max_steps` cap (default 1000) to prevent an unattended recording from running infinitely.
- **Safe Discard**: If an episode is discarded, the file handle is explicitly closed in a `try/except` block before the file is deleted, handling OS-level locking gracefully.

---

## 2. Input Validation (Data Trust Boundary)

Before any recorded data is allowed into the training loop, it must be validated. `src.data.validator` acts as the trust boundary.

### Protections in `validate_episode_file`
- **Strict Schema Checks**: Every JSON object is checked against `TRANSITION_REQUIRED_FIELDS`. Missing keys cause the line to be rejected.
- **Action Validation**: The `action` field is explicitly checked against the allowed `0-4` range. A malicious file injecting `action=99` to crash the env will fail validation.
- **State Shape Consistency**: The length of the continuous state vector is inferred from the first line and enforced on all subsequent lines. This prevents tensor shape mismatches during PyTorch training later.
- **Type Checking**: Rewards are explicitly checked to be `int` or `float`.

---

## 3. UI Input Safety (Keyboard Controller)

The `KeyboardController` maps raw strings from the Streamlit UI to environment actions.

### Protections in `KeyboardController`
- **Default Action**: If a user mashes the keyboard or an unexpected key string is sent from the UI, the controller safely falls back to a default valid action (usually `IDLE` / `1`), rather than throwing an exception that kills the UI.
- **Action Validation**: Even custom key mappings are range-checked upon creation to ensure no invalid actions can be bound.

---

## 4. No Dangerous Constructs

Per the project rules, the following are strictly prohibited and **not used** anywhere in the data layer:
- `pickle` (subject to arbitrary code execution; PyTorch `torch.load(weights_only=True)` will be used later for models).
- `eval()` or `exec()` for parsing data. All data is parsed via Python's standard `json` module.
- Bare `except:` blocks that swallow keyboard interrupts or mask underlying validation bugs.

*Last updated: Build Step 2*
