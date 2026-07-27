"""Deterministic six-region pair construction for Go2 TRACE labels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .go2_feedback_prompt import build_go2_feedback_prompt
from .schemas import PAIR_SCHEMA_VERSION, SUMMARY_SCHEMA_HASH


@dataclass(frozen=True)
class PairQuota:
    within_region: int
    cross_region: int

    def __post_init__(self) -> None:
        if self.within_region < 0 or self.cross_region < 0:
            raise ValueError("Pair quotas must be non-negative.")
        if self.within_region + self.cross_region < 1:
            raise ValueError("At least one feedback pair is required.")


COMMAND_REGIONS = ("front", "back", "left", "right", "pure_yaw", "stand")


def command_region(
    summary: Mapping[str, Any],
    *,
    planar_command_scales: Sequence[float],
) -> str:
    """Assign front/back/left/right by the two diagonals in normalized vx-vy."""

    mode = str(summary.get("command_mode", ""))
    if mode == "stand":
        return "stand"
    if mode == "pure_yaw":
        return "pure_yaw"
    if mode not in {"pure_x", "pure_y", "xy", "x_yaw", "y_yaw", "xy_yaw"}:
        raise ValueError(f"Unknown command_mode for region pairing: {mode!r}.")
    scales = tuple(map(float, planar_command_scales))
    if len(scales) != 2 or any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("planar_command_scales must contain two finite positive values.")
    command_mean = summary.get("command_mean")
    if not isinstance(command_mean, Mapping):
        raise ValueError("Feedback summary is missing signed command_mean metadata.")
    x = float(command_mean.get("vx", float("nan"))) / scales[0]
    y = float(command_mean.get("vy", float("nan"))) / scales[1]
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("Feedback summary contains a non-finite planar command mean.")
    # The boundaries are the two diagonal rays y=x and y=-x.  Ties are
    # assigned deterministically to the front/back wedges.
    boundary_tolerance = 1e-6
    if x + boundary_tolerance >= abs(y):
        return "front"
    if x - boundary_tolerance <= -abs(y):
        return "back"
    return "left" if y > 0.0 else "right"


def _stable_pair_key(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    ids = sorted((str(left.get("trajectory_id", "")), str(right.get("trajectory_id", ""))))
    return hashlib.sha256("\0".join(ids).encode("utf-8")).hexdigest()


def _choose(
    candidates: Sequence[tuple[int, int]],
    count: int,
    *,
    rng: np.random.Generator,
    kind: str,
) -> list[tuple[int, int]]:
    if len(candidates) < count:
        raise ValueError(
            f"Requested {count} {kind} pairs but only {len(candidates)} unique pairs exist."
        )
    if count == 0:
        return []
    order = rng.permutation(len(candidates))[:count]
    return [candidates[int(index)] for index in order]


def build_feedback_pairs(
    summaries: Sequence[Mapping[str, Any]],
    *,
    quota: PairQuota,
    pair_prefix: str,
    seed: int,
    planar_command_scales: Sequence[float] = (0.5, 0.2),
) -> list[dict[str, Any]]:
    """Sample exact 80/20-style within/cross-region quotas without a scorer."""

    if len(summaries) < 2:
        raise ValueError("At least two summaries are needed.")
    normalized = [dict(item) for item in summaries]
    for item in normalized:
        if item.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            raise ValueError("Every feedback candidate must use the current summary schema.")
        if not str(item.get("trajectory_id", "")):
            raise ValueError("Every feedback candidate needs a non-empty trajectory_id.")
    regions = [
        command_region(item, planar_command_scales=planar_command_scales)
        for item in normalized
    ]
    within: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            pair = (left, right)
            if regions[left] == regions[right]:
                within.append(pair)
            else:
                cross.append(pair)
    for candidates in (within, cross):
        candidates.sort(
            key=lambda pair: _stable_pair_key(normalized[pair[0]], normalized[pair[1]])
        )
    rng = np.random.default_rng(int(seed))
    selected = [
        *[(left, right, "within_command_region") for left, right in _choose(
            within, quota.within_region, rng=rng, kind="within-region"
        )],
        *[(left, right, "cross_command_region") for left, right in _choose(
            cross, quota.cross_region, rng=rng, kind="cross-region"
        )],
    ]
    rng.shuffle(selected)

    rows: list[dict[str, Any]] = []
    for index, (left, right, comparison_type) in enumerate(selected):
        row: dict[str, Any] = {
            "pair_schema_version": PAIR_SCHEMA_VERSION,
            "pair_id": f"{pair_prefix}_{index:06d}",
            "comparison_type": comparison_type,
            "command_region_i": regions[left],
            "command_region_j": regions[right],
            "trajectory_i": normalized[left],
            "trajectory_j": normalized[right],
        }
        row["prompt"] = build_go2_feedback_prompt(row)
        rows.append(row)
    counts = {
        "within_command_region": sum(
            row["comparison_type"] == "within_command_region" for row in rows
        ),
        "cross_command_region": sum(
            row["comparison_type"] == "cross_command_region" for row in rows
        ),
    }
    if (
        counts["within_command_region"] != quota.within_region
        or counts["cross_command_region"] != quota.cross_region
    ):
        raise RuntimeError("Pair construction did not preserve the configured exact quota.")
    return rows
