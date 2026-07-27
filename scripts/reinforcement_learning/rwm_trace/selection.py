"""Deterministic global, command-mode, and six-region TRACE selectors."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import SUMMARY_SCHEMA_HASH
from .feedback_pairs import COMMAND_REGIONS, command_region


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    valid_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    alpha: float
    selected_count: int
    ranking: tuple[int, ...]


COMMAND_MODE_ORDER: tuple[str, ...] = (
    "stand",
    "pure_x",
    "pure_y",
    "pure_yaw",
    "xy",
    "x_yaw",
    "y_yaw",
    "xy_yaw",
)


def command_mode(summary: Mapping[str, Any]) -> str:
    active = summary.get("command_active")
    if not isinstance(active, Mapping):
        raise ValueError("TRACE summary lacks command_active mode metadata.")
    key = tuple(bool(active.get(axis, False)) for axis in ("vx", "vy", "yaw"))
    names = {
        (False, False, False): "stand",
        (True, False, False): "pure_x",
        (False, True, False): "pure_y",
        (False, False, True): "pure_yaw",
        (True, True, False): "xy",
        (True, False, True): "x_yaw",
        (False, True, True): "y_yaw",
        (True, True, True): "xy_yaw",
    }
    return names[key]


def _tie_key(summary: Mapping[str, Any], seed: int) -> str:
    identity = str(summary.get("trajectory_id", ""))
    return hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()


def global_top_alpha(
    summaries: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    seed: int,
    valid_mask: Sequence[bool] | None = None,
) -> SelectionResult:
    """Select ``ceil(alpha * N_valid)`` once over the complete candidate pool."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0,1].")
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(score) != len(summaries):
        raise ValueError("Score count does not match summary count.")
    if valid_mask is None:
        valid = np.ones(len(summaries), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(valid) != len(summaries):
            raise ValueError("valid_mask count does not match summary count.")
    for index, summary in enumerate(summaries):
        if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            valid[index] = False
        if not str(summary.get("trajectory_id", "")) or not np.isfinite(score[index]):
            valid[index] = False
    valid_indices = np.flatnonzero(valid).tolist()
    if not valid_indices:
        raise ValueError("No valid trajectory remains for learned-score selection.")
    ranking = sorted(
        valid_indices,
        key=lambda index: (
            -float(score[index]),
            _tie_key(summaries[index], seed),
        ),
    )
    selected_count = int(math.ceil(float(alpha) * len(valid_indices)))
    selected = ranking[:selected_count]
    selected_set = set(selected)
    rejected = [index for index in range(len(summaries)) if index not in selected_set]
    return SelectionResult(
        selected_indices=tuple(selected),
        valid_indices=tuple(valid_indices),
        rejected_indices=tuple(rejected),
        alpha=float(alpha),
        selected_count=selected_count,
        ranking=tuple(ranking),
    )


def command_stratified_top_alpha(
    summaries: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    seed: int,
    valid_mask: Sequence[bool] | None = None,
) -> SelectionResult:
    """Apply top-alpha independently inside each of the eight command modes."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0,1].")
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(score) != len(summaries):
        raise ValueError("Score count does not match summary count.")
    if valid_mask is None:
        valid = np.ones(len(summaries), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(valid) != len(summaries):
            raise ValueError("valid_mask count does not match summary count.")
    modes: list[str | None] = []
    for index, summary in enumerate(summaries):
        if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            valid[index] = False
        if not str(summary.get("trajectory_id", "")) or not np.isfinite(score[index]):
            valid[index] = False
        try:
            modes.append(command_mode(summary))
        except ValueError:
            modes.append(None)
            valid[index] = False

    selected: list[int] = []
    ranking: list[int] = []
    for mode in COMMAND_MODE_ORDER:
        members = [
            index
            for index in range(len(summaries))
            if valid[index] and modes[index] == mode
        ]
        mode_ranking = sorted(
            members,
            key=lambda index: (
                -float(score[index]),
                _tie_key(summaries[index], seed),
            ),
        )
        ranking.extend(mode_ranking)
        selected.extend(mode_ranking[: int(math.ceil(float(alpha) * len(members)))])

    valid_indices = np.flatnonzero(valid).tolist()
    if not valid_indices:
        raise ValueError("No valid trajectory remains for command-stratified selection.")
    selected_set = set(selected)
    rejected = [
        index for index in range(len(summaries)) if index not in selected_set
    ]
    return SelectionResult(
        selected_indices=tuple(selected),
        valid_indices=tuple(valid_indices),
        rejected_indices=tuple(rejected),
        alpha=float(alpha),
        selected_count=len(selected),
        ranking=tuple(ranking),
    )


def command_region_stratified_top_alpha(
    summaries: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    seed: int,
    valid_mask: Sequence[bool] | None = None,
    planar_command_scales: Sequence[float] = (0.5, 0.2),
) -> SelectionResult:
    """Apply top-alpha independently in the six directional command regions."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0,1].")
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(score) != len(summaries):
        raise ValueError("Score count does not match summary count.")
    valid = (
        np.ones(len(summaries), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool).reshape(-1)
    )
    if len(valid) != len(summaries):
        raise ValueError("valid_mask count does not match summary count.")
    regions: list[str | None] = []
    for index, summary in enumerate(summaries):
        if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            valid[index] = False
        if not str(summary.get("trajectory_id", "")) or not np.isfinite(score[index]):
            valid[index] = False
        try:
            regions.append(
                command_region(
                    summary,
                    planar_command_scales=planar_command_scales,
                )
            )
        except ValueError:
            regions.append(None)
            valid[index] = False

    selected: list[int] = []
    ranking: list[int] = []
    for region in COMMAND_REGIONS:
        members = [
            index
            for index in range(len(summaries))
            if valid[index] and regions[index] == region
        ]
        region_ranking = sorted(
            members,
            key=lambda index: (
                -float(score[index]),
                _tie_key(summaries[index], seed),
            ),
        )
        ranking.extend(region_ranking)
        selected.extend(
            region_ranking[: int(math.ceil(float(alpha) * len(members)))]
        )

    valid_indices = np.flatnonzero(valid).tolist()
    if not valid_indices:
        raise ValueError("No valid trajectory remains for command-region selection.")
    selected_set = set(selected)
    return SelectionResult(
        selected_indices=tuple(selected),
        valid_indices=tuple(valid_indices),
        rejected_indices=tuple(
            index for index in range(len(summaries)) if index not in selected_set
        ),
        alpha=float(alpha),
        selected_count=len(selected),
        ranking=tuple(ranking),
    )
