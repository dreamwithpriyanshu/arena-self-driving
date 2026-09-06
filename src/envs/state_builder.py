"""
State builder — transforms raw HighwayEnv observations into the
representations consumed by the tabular SARSA policy.

Provides a continuous flat vector (retained for recorded data) and a
discrete state index (used by the Q-table).  Both are derived from
the same environment step observation.

Layer: envs  (knows nothing about agents, training, or Streamlit)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default_env.yaml"


def _load_sarsa_bins(config_path: Optional[str | Path] = None) -> dict[str, list[float]]:
    """Load SARSA bin boundaries from the YAML config."""
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found at {config_path} — cannot load SARSA bins."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("sarsa_bins", {})


# ---------------------------------------------------------------------------
# Raw recorded state (continuous flat vector)
# ---------------------------------------------------------------------------

def _default_vehicles_count() -> int:
    """Return the configured default observation vehicles_count from YAML.

    Falls back to 6 if config cannot be read. This helps when expanding
    1-D ego-only observations into a (V, F) matrix.
    """
    try:
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        obs_cfg = cfg.get("observation", {})
        return int(obs_cfg.get("vehicles_count", cfg.get("vehicles_count", 6)))
    except Exception:
        return 6


def build_raw_state(obs: np.ndarray) -> np.ndarray:
    """
    Flatten the Kinematics observation into a 1-D float32 vector.

    Parameters
    ----------
    obs : np.ndarray
        Either shape ``(V, F)`` from HighwayEnv's Kinematics observation
        (V = vehicles_count, F = features per vehicle), or a 1-D ego-only
        feature vector of shape ``(F,)``. If a 1-D vector is provided it is
        expanded into a ``(V, F)`` matrix by placing the ego row at index 0
        and zero-padding neighbour rows.

    Returns
    -------
    np.ndarray
        1-D vector of shape ``(V * F,)`` with dtype float32.
    """
    obs = np.asarray(obs, dtype=np.float32)

    # Accept both full (V, F) matrices and ego-only (F,) vectors.
    # When a 1-D vector is received, neighbour rows are zero-padded which
    # means all neighbour features (gap, speed-diff, lane-occupancy) will
    # read as zero — a conservative "no neighbours visible" assumption.
    if obs.ndim == 1:
        features = obs.size
        vehicles = _default_vehicles_count()
        logger.warning(
            "build_raw_state received 1-D obs of shape (%d,); "
            "expanding to (%d, %d) with zero-padded neighbour rows. "
            "Neighbour information will be absent.",
            features, vehicles, features,
        )
        mat = np.zeros((vehicles, features), dtype=np.float32)
        mat[0, :features] = obs
        obs = mat

    if obs.ndim != 2:
        raise ValueError(
            f"Expected a 2-D observation (V, F), got shape {obs.shape}."
        )
    return obs.flatten()


def raw_state_dim(vehicles_count: int = 6, features_count: int = 6) -> int:
    """Return the expected dimensionality of the raw recorded state vector."""
    return vehicles_count * features_count


def build_raw_state_from_env(env: Any, vehicles_count: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Best-effort reconstruction of a full (V, F) kinematics matrix from the
    underlying environment object. This inspects `env.unwrapped` for road and
    vehicle objects and converts them into rows of [x, y, vx, vy, cos_h, sin_h]
    relative to ego.

    Returns None if reconstruction is not possible.
    """
    try:
        unwrapped = getattr(env, "unwrapped", env)
        road = getattr(unwrapped, "road", None)
        if road is None:
            logger.warning(
                "build_raw_state_from_env: env.unwrapped has no 'road' "
                "attribute — reconstruction not possible."
            )
            return None

        # Collect vehicles from the road object. `road.vehicles` may be a
        # dict-like or list-like; handle common cases.
        vehicles = []
        if hasattr(road, "vehicles"):
            vs = getattr(road, "vehicles")
            try:
                vehicles = list(vs)
            except Exception:
                # Attempt dict values
                try:
                    vehicles = list(vs.values())
                except Exception:
                    vehicles = []

        if not vehicles:
            logger.warning(
                "build_raw_state_from_env: road.vehicles is empty — "
                "reconstruction not possible."
            )
            return None

        ego = getattr(unwrapped, "vehicle", vehicles[0])

        # Build rows: attempt to read position, speed, and heading from each
        # vehicle object. This is a best-effort conversion that won't raise on
        # missing attributes.
        rows = []
        for v in vehicles:
            try:
                pos = getattr(v, "position", None)
                if pos is None:
                    # Some environments expose `x`, `y` directly
                    x = float(getattr(v, "x", 0.0))
                    y = float(getattr(v, "y", 0.0))
                else:
                    ego_pos = getattr(ego, "position", (0.0, 0.0))
                    x = float(pos[0] - ego_pos[0])
                    y = float(pos[1] - ego_pos[1])

                speed = float(getattr(v, "speed", getattr(v, "velocity", 0.0)))
                # vy not always available; attempt to use velocity vector
                vel = getattr(v, "velocity", None)
                if vel is None:
                    vx = speed
                    vy = 0.0
                else:
                    try:
                        vx = float(vel[0])
                        vy = float(vel[1])
                    except Exception:
                        vx = speed
                        vy = 0.0

                heading = float(getattr(v, "heading", getattr(v, "angle", 0.0)))
                cos_h = float(np.cos(heading))
                sin_h = float(np.sin(heading))

                rows.append([x, y, vx, vy, cos_h, sin_h])
            except Exception:
                logger.warning(
                    "build_raw_state_from_env: failed to extract "
                    "features from vehicle %d; zero-padding row.",
                    len(rows),
                )
                rows.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        vc = vehicles_count or _default_vehicles_count()
        features = 6
        mat = np.zeros((vc, features), dtype=np.float32)
        for i, r in enumerate(rows[:vc]):
            mat[i, :len(r)] = r
        if len(rows) < vc:
            logger.warning(
                "build_raw_state_from_env: reconstructed %d of %d vehicle rows; "
                "zero-padding the remaining %d row(s).",
                len(rows), vc, vc - len(rows),
            )
        return mat
    except Exception:
        logger.warning(
            "build_raw_state_from_env: unexpected error during "
            "reconstruction — returning None.",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Discrete state for SARSA  (integer index)
# ---------------------------------------------------------------------------

# Feature-index constants for the default Kinematics observation.
# Each row: [x, y, vx, vy, cos_h, sin_h]  (relative to ego when absolute=False).
_X, _Y, _VX, _VY = 0, 1, 2, 3

# Ego is always row 0; neighbours are rows 1..V-1.
_EGO = 0


def _digitise(value: float, boundaries: list[float]) -> int:
    """
    Return the bin index for *value* given sorted *boundaries*.

    Bins:
      - 0              if value < boundaries[0]
      - 1..(N-1)       for intermediate ranges
      - N              if value >= boundaries[-1]

    where N = len(boundaries).
    """
    return int(np.digitize(value, boundaries))


def _find_nearest_front(obs: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the row of the nearest vehicle *ahead* of ego (positive x).

    Returns ``None`` if no vehicle is ahead.
    """
    neighbours = obs[1:]  # skip ego
    ahead_mask = neighbours[:, _X] > 0
    if not np.any(ahead_mask):
        return None
    ahead = neighbours[ahead_mask]
    nearest_idx = np.argmin(ahead[:, _X])
    return ahead[nearest_idx]


def _find_nearest_rear(obs: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the row of the nearest vehicle *behind* ego (negative x).

    Returns ``None`` if no vehicle is behind.
    """
    neighbours = obs[1:]
    behind_mask = neighbours[:, _X] < 0
    if not np.any(behind_mask):
        return None
    behind = neighbours[behind_mask]
    nearest_idx = np.argmax(behind[:, _X])  # closest to 0 among negatives
    return behind[nearest_idx]


def _is_adjacent_lane_free(
    obs: np.ndarray,
    direction: float,
    lane_threshold: float = 0.15,
    gap_threshold: float = 0.15,
) -> bool:
    """
    Check whether the lane to *direction* (negative = left, positive = right)
    of ego appears free of nearby vehicles.

    Uses normalised y-position: a neighbour is "in the adjacent lane" if
    its relative y is in the right range and its x-gap is small enough to
    be dangerous.
    """
    neighbours = obs[1:]
    for row in neighbours:
        # Check if the presence flag is active (row not all-zero)
        if np.allclose(row, 0.0):
            continue
        y_rel = row[_Y]
        x_gap = abs(row[_X])
        # Vehicle is roughly in the adjacent lane in *direction*
        in_lane = (direction * y_rel > 0) and abs(y_rel) < lane_threshold
        close_enough = x_gap < gap_threshold
        if in_lane and close_enough:
            return False
    return True


def build_discrete_state(
    obs: np.ndarray,
    sarsa_bins: Optional[dict[str, list[float]]] = None,
    lanes_count: int = 4,
) -> int:
    """
    Convert a Kinematics observation into a single integer state index
    suitable for tabular SARSA.

    The categorical features are:

    ========  ======  ===================================================
    Feature   #Bins   Description
    ========  ======  ===================================================
    Lane        4     Ego vehicle's lane (0..lanes_count-1)
    Speed       3     Ego speed bucket (slow / normal / fast)
    Front gap   3     Gap to nearest vehicle ahead (close / medium / far)
    Front Δv    3     Speed difference with front vehicle
    Left free   2     Is the left lane free of nearby vehicles?
    Right free  2     Is the right lane free of nearby vehicles?
    Rear gap    2     Gap to nearest vehicle behind (close / safe)
    ========  ======  ===================================================

    Total states (default): 4 × 3 × 3 × 3 × 2 × 2 × 2 = **864**

    Parameters
    ----------
    obs : np.ndarray
        Shape ``(V, F)`` Kinematics observation (normalised, relative).
    sarsa_bins : dict, optional
        Bin boundaries loaded from config.  If ``None``, loaded from
        ``configs/default_env.yaml``.
    lanes_count : int
        Number of highway lanes (used for lane feature size).

    Returns
    -------
    int
        A non-negative integer uniquely identifying the discrete state.
    """
    if sarsa_bins is None:
        sarsa_bins = _load_sarsa_bins()

    obs = np.asarray(obs, dtype=np.float32)
    # Support ego-only 1-D observations by expanding to (V, F)
    if obs.ndim == 1:
        features = obs.size
        vehicles = _default_vehicles_count()
        logger.warning(
            "build_discrete_state received 1-D obs of shape (%d,); "
            "expanding to (%d, %d) with zero-padded neighbour rows.",
            features, vehicles, features,
        )
        mat = np.zeros((vehicles, features), dtype=np.float32)
        mat[0, :features] = obs
        obs = mat

    if obs.ndim != 2:
        raise ValueError(
            f"Expected a 2-D observation (V, F), got shape {obs.shape}."
        )

    ego = obs[_EGO]
    speed_bins = sarsa_bins.get("speed_bins", [0.33, 0.66])
    front_gap_bins = sarsa_bins.get("front_gap_bins", [0.2, 0.5])
    front_speed_bins = sarsa_bins.get("front_speed_diff_bins", [-0.1, 0.1])
    rear_gap_bins = sarsa_bins.get("rear_gap_bins", [0.3])

    # 1) Lane index  (clamp to valid range)
    #    In normalised-relative mode the ego y is always 0, so we use the
    #    *absolute* ego lane from info or approximate via y.  For robustness
    #    we discretise the ego vx as a proxy for the lane ID when absolute
    #    positioning isn't available.  HighwayEnv with normalize+relative
    #    still provides the ego's y ≈ 0, so we fall back to the y of the
    #    first feature row.  The sim façade will provide the true lane later;
    #    for now we estimate from the observation.
    #    With absolute=False and normalize=True, ego features are all 0.
    #    We'll get the true lane from the env info dict in env_manager.
    lane_idx = 0  # placeholder — overridden by env_manager with true lane

    # 2) Speed bucket
    ego_speed = float(ego[_VX])
    speed_cat = _digitise(ego_speed, speed_bins)

    # 3) Front gap
    front = _find_nearest_front(obs)
    if front is not None:
        front_gap = float(front[_X])
        front_gap_cat = _digitise(front_gap, front_gap_bins)
        # 4) Front speed diff
        front_vx_diff = float(front[_VX])
        front_vdiff_cat = _digitise(front_vx_diff, front_speed_bins)
    else:
        front_gap_cat = len(front_gap_bins)   # "far" — no vehicle ahead
        front_vdiff_cat = len(front_speed_bins)  # "faster" — no obstacle

    # 5) Left lane free?
    left_free = int(_is_adjacent_lane_free(obs, direction=-1.0))

    # 6) Right lane free?
    right_free = int(_is_adjacent_lane_free(obs, direction=1.0))

    # 7) Rear gap
    rear = _find_nearest_rear(obs)
    if rear is not None:
        rear_gap = abs(float(rear[_X]))
        rear_gap_cat = _digitise(rear_gap, rear_gap_bins)
    else:
        rear_gap_cat = len(rear_gap_bins)  # "safe" — nobody behind

    # Combine into a single index using mixed-radix encoding.
    sizes = [
        lanes_count,                         # lane
        len(speed_bins) + 1,                 # speed
        len(front_gap_bins) + 1,             # front gap
        len(front_speed_bins) + 1,           # front Δv
        2,                                   # left free
        2,                                   # right free
        len(rear_gap_bins) + 1,              # rear gap
    ]
    values = [
        lane_idx,
        speed_cat,
        front_gap_cat,
        front_vdiff_cat,
        left_free,
        right_free,
        rear_gap_cat,
    ]

    state_index = 0
    multiplier = 1
    for val, size in zip(reversed(values), reversed(sizes)):
        state_index += val * multiplier
        multiplier *= size

    return state_index


def build_discrete_state_with_lane(
    obs: np.ndarray,
    lane_index: int,
    sarsa_bins: Optional[dict[str, list[float]]] = None,
    lanes_count: int = 4,
) -> int:
    """
    Same as :func:`build_discrete_state` but accepts the *true* ego lane
    index from the environment info dict, replacing the placeholder.

    This is the preferred entry point when the simulation façade provides
    lane information alongside the observation.
    """
    if sarsa_bins is None:
        sarsa_bins = _load_sarsa_bins()

    obs = np.asarray(obs, dtype=np.float32)
    # Support ego-only 1-D observations by expanding to (V, F)
    if obs.ndim == 1:
        features = obs.size
        vehicles = _default_vehicles_count()
        logger.warning(
            "build_discrete_state_with_lane received 1-D obs of shape "
            "(%d,); expanding to (%d, %d) with zero-padded neighbour rows.",
            features, vehicles, features,
        )
        mat = np.zeros((vehicles, features), dtype=np.float32)
        mat[0, :features] = obs
        obs = mat

    if obs.ndim != 2:
        raise ValueError(
            f"Expected a 2-D observation (V, F), got shape {obs.shape}."
        )

    ego = obs[_EGO]
    speed_bins = sarsa_bins.get("speed_bins", [0.33, 0.66])
    front_gap_bins = sarsa_bins.get("front_gap_bins", [0.2, 0.5])
    front_speed_bins = sarsa_bins.get("front_speed_diff_bins", [-0.1, 0.1])
    rear_gap_bins = sarsa_bins.get("rear_gap_bins", [0.3])

    lane_idx = max(0, min(lane_index, lanes_count - 1))
    speed_cat = _digitise(float(ego[_VX]), speed_bins)

    front = _find_nearest_front(obs)
    if front is not None:
        front_gap_cat = _digitise(float(front[_X]), front_gap_bins)
        front_vdiff_cat = _digitise(float(front[_VX]), front_speed_bins)
    else:
        front_gap_cat = len(front_gap_bins)
        front_vdiff_cat = len(front_speed_bins)

    left_free = int(_is_adjacent_lane_free(obs, direction=-1.0))
    right_free = int(_is_adjacent_lane_free(obs, direction=1.0))

    rear = _find_nearest_rear(obs)
    if rear is not None:
        rear_gap_cat = _digitise(abs(float(rear[_X])), rear_gap_bins)
    else:
        rear_gap_cat = len(rear_gap_bins)

    sizes = [
        lanes_count,
        len(speed_bins) + 1,
        len(front_gap_bins) + 1,
        len(front_speed_bins) + 1,
        2,
        2,
        len(rear_gap_bins) + 1,
    ]
    values = [
        lane_idx,
        speed_cat,
        front_gap_cat,
        front_vdiff_cat,
        left_free,
        right_free,
        rear_gap_cat,
    ]

    state_index = 0
    multiplier = 1
    for val, size in zip(reversed(values), reversed(sizes)):
        state_index += val * multiplier
        multiplier *= size

    return state_index


def total_discrete_states(
    lanes_count: int = 4,
    sarsa_bins: Optional[dict[str, list[float]]] = None,
) -> int:
    """Return the total number of discrete states (for SARSA Q-table sizing)."""
    if sarsa_bins is None:
        sarsa_bins = _load_sarsa_bins()

    speed_bins = sarsa_bins.get("speed_bins", [0.33, 0.66])
    front_gap_bins = sarsa_bins.get("front_gap_bins", [0.2, 0.5])
    front_speed_bins = sarsa_bins.get("front_speed_diff_bins", [-0.1, 0.1])
    rear_gap_bins = sarsa_bins.get("rear_gap_bins", [0.3])

    total = (
        lanes_count
        * (len(speed_bins) + 1)
        * (len(front_gap_bins) + 1)
        * (len(front_speed_bins) + 1)
        * 2  # left free
        * 2  # right free
        * (len(rear_gap_bins) + 1)
    )
    return total
