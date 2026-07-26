"""One global top-alpha selector shared by rule bootstrap and learned TRACE."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import SUMMARY_SCHEMA_HASH


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    valid_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    alpha: float
    selected_count: int
    ranking: tuple[int, ...]


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
