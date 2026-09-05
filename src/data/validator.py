"""
Data validator — checks JSONL episode files for structural integrity.

Before any human demonstration file is trusted by a trainer, it must
pass validation:

1. **Schema check** — every transition line has the required fields.
2. **Action check** — every action index is in the valid set (0–4).
3. **Continuity check** — step indices are sequential (0, 1, 2, …).
4. **State-shape check** — all state vectors have consistent length.
5. **Metadata check** — the first line is valid episode metadata.

Layer: data  (depends on schemas and envs.actions; no side effects beyond logging)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data.schemas import TRANSITION_REQUIRED_FIELDS, EpisodeMetadata, Transition
from src.envs.actions import is_valid_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validating a JSONL episode file."""

    is_valid: bool = True
    """True if all checks passed."""

    filepath: str = ""
    """Path to the validated file."""

    num_transitions: int = 0
    """Number of valid transition lines found."""

    errors: list[str] = field(default_factory=list)
    """Human-readable error messages for each issue found."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal issues."""

    metadata: Optional[EpisodeMetadata] = None
    """Parsed episode metadata (if the header line was valid)."""

    def add_error(self, msg: str) -> None:
        """Record an error and mark the result as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(msg)

    def summary(self) -> str:
        """One-line summary of the validation outcome."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"[{status}] {self.filepath}: "
            f"{self.num_transitions} transitions, "
            f"{len(self.errors)} errors, "
            f"{len(self.warnings)} warnings"
        )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_episode_file(
    filepath: str | Path,
    expected_state_dim: Optional[int] = None,
) -> ValidationResult:
    """
    Validate a single JSONL episode file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.jsonl`` file.
    expected_state_dim : int, optional
        If provided, every state vector must have this length.
        If ``None``, the dimension is inferred from the first transition.

    Returns
    -------
    ValidationResult
        Detailed validation outcome.
    """
    filepath = Path(filepath)
    result = ValidationResult(filepath=str(filepath))

    if not filepath.exists():
        result.add_error(f"File does not exist: {filepath}")
        return result

    if not filepath.suffix == ".jsonl":
        result.add_warning(f"File extension is '{filepath.suffix}', expected '.jsonl'")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as exc:
        result.add_error(f"Cannot read file: {exc}")
        return result

    if not lines:
        result.add_error("File is empty")
        return result

    # ---- Line 1: metadata ----
    try:
        meta_raw = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        result.add_error(f"Line 1 is not valid JSON: {exc}")
        return result

    if meta_raw.get("_type") != "episode_metadata":
        result.add_error(
            f"Line 1 missing '_type': 'episode_metadata' "
            f"(got _type={meta_raw.get('_type')!r})"
        )
        return result

    try:
        result.metadata = EpisodeMetadata.from_dict(meta_raw)
    except (TypeError, ValueError) as exc:
        result.add_error(f"Invalid metadata: {exc}")
        return result

    # ---- Lines 2+: transitions ----
    inferred_state_dim: Optional[int] = expected_state_dim
    expected_step = 0

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue  # skip blank lines

        # Parse JSON
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            result.add_error(f"Line {line_num}: invalid JSON — {exc}")
            continue

        # Schema check
        missing = TRANSITION_REQUIRED_FIELDS - set(raw.keys())
        if missing:
            result.add_error(
                f"Line {line_num}: missing required fields: {sorted(missing)}"
            )
            continue

        # Action check
        action = raw.get("action")
        if not is_valid_action(action):
            result.add_error(
                f"Line {line_num}: invalid action {action!r} "
                f"(expected 0–4)"
            )

        # Step continuity
        step = raw.get("step")
        if isinstance(step, int):
            if step != expected_step:
                result.add_warning(
                    f"Line {line_num}: step={step}, expected {expected_step} "
                    f"(non-sequential)"
                )
            expected_step = step + 1

        # State shape consistency
        state = raw.get("state")
        if isinstance(state, list):
            if inferred_state_dim is None:
                inferred_state_dim = len(state)
            elif len(state) != inferred_state_dim:
                result.add_error(
                    f"Line {line_num}: state length {len(state)} != "
                    f"expected {inferred_state_dim}"
                )

        next_state = raw.get("next_state")
        if isinstance(next_state, list) and inferred_state_dim is not None:
            if len(next_state) != inferred_state_dim:
                result.add_error(
                    f"Line {line_num}: next_state length {len(next_state)} != "
                    f"expected {inferred_state_dim}"
                )

        # Reward type check
        reward = raw.get("reward")
        if not isinstance(reward, (int, float)):
            result.add_error(
                f"Line {line_num}: reward is not numeric ({type(reward).__name__})"
            )

        result.num_transitions += 1

    # Cross-check metadata
    if result.metadata is not None and result.metadata.num_transitions > 0:
        if result.num_transitions != result.metadata.num_transitions:
            result.add_warning(
                f"Metadata says {result.metadata.num_transitions} transitions "
                f"but file has {result.num_transitions}"
            )

    if result.num_transitions == 0 and result.is_valid:
        result.add_warning("File has metadata but zero transitions")

    logger.info(result.summary())
    return result


def validate_directory(
    directory: str | Path,
    expected_state_dim: Optional[int] = None,
) -> list[ValidationResult]:
    """
    Validate all ``.jsonl`` files in a directory.

    Returns a list of :class:`ValidationResult` objects, one per file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return []

    results = []
    for fpath in sorted(directory.glob("*.jsonl")):
        results.append(validate_episode_file(fpath, expected_state_dim))

    valid = sum(1 for r in results if r.is_valid)
    logger.info(
        "Validated %d files in %s: %d valid, %d invalid",
        len(results), directory, valid, len(results) - valid,
    )
    return results
