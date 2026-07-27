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


def region_quota_top_alpha(
    summaries: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | np.ndarray,
    *,
    regions: Sequence[str],
    target_weights: Mapping[str, float],
    quota_fraction: float,
    alpha: float,
    seed: int,
    valid_mask: Sequence[bool] | None = None,
) -> SelectionResult:
    """Reserve a soft region-balanced prefix, then fill by global score."""

    base = global_top_alpha(
        summaries,
        scores,
        alpha=alpha,
        seed=seed,
        valid_mask=valid_mask,
    )
    if len(regions) != len(summaries):
        raise ValueError("Region count does not match summary count.")
    if not 0.0 <= float(quota_fraction) <= 1.0:
        raise ValueError("quota_fraction must be in [0,1].")
    weights = {
        str(region): float(weight)
        for region, weight in target_weights.items()
    }
    if not weights or any(
        not np.isfinite(weight) or weight <= 0.0
        for weight in weights.values()
    ):
        raise ValueError("Region target weights must be finite and positive.")
    unknown = set(regions) - set(weights)
    if unknown:
        raise ValueError(f"Missing target weights for regions: {sorted(unknown)}")

    quota_count = int(math.floor(base.selected_count * float(quota_fraction)))
    if quota_count == 0:
        return base
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_set = set(base.valid_indices)
    queues = {
        region: sorted(
            [
                index
                for index in base.valid_indices
                if str(regions[index]) == region
            ],
            key=lambda index: (
                -float(score[index]),
                _tie_key(summaries[index], seed),
            ),
        )
        for region in weights
    }
    selected: list[int] = []
    selected_by_region = {region: 0 for region in weights}
    while len(selected) < quota_count:
        available = [region for region, queue in queues.items() if queue]
        if not available:
            break
        region = min(
            available,
            key=lambda name: (
                selected_by_region[name] / weights[name],
                name,
            ),
        )
        selected.append(queues[region].pop(0))
        selected_by_region[region] += 1

    selected_set = set(selected)
    for index in base.ranking:
        if len(selected) >= base.selected_count:
            break
        if index in valid_set and index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    rejected = [
        index for index in range(len(summaries)) if index not in selected_set
    ]
    return SelectionResult(
        selected_indices=tuple(selected),
        valid_indices=base.valid_indices,
        rejected_indices=tuple(rejected),
        alpha=base.alpha,
        selected_count=len(selected),
        ranking=base.ranking,
    )


def lateral_symmetric_top_alpha(
    summaries: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | np.ndarray,
    *,
    regions: Sequence[str],
    balance_fraction: float,
    alpha: float,
    seed: int,
    valid_mask: Sequence[bool] | None = None,
) -> SelectionResult:
    """Reserve a score-ranked, sign-symmetric lateral prefix."""

    base = global_top_alpha(
        summaries,
        scores,
        alpha=alpha,
        seed=seed,
        valid_mask=valid_mask,
    )
    if len(regions) != len(summaries):
        raise ValueError("Region count does not match summary count.")
    if not 0.0 <= float(balance_fraction) <= 1.0:
        raise ValueError("Lateral balance fraction must be in [0,1].")
    reserve = int(math.floor(base.selected_count * float(balance_fraction)))
    reserve -= reserve % 2
    if reserve == 0:
        return base
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    queues = {
        region: sorted(
            [
                index
                for index in base.valid_indices
                if str(regions[index]) == region
            ],
            key=lambda index: (
                -float(score[index]),
                _tie_key(summaries[index], seed),
            ),
        )
        for region in ("left", "right")
    }
    per_sign = min(reserve // 2, *(len(queue) for queue in queues.values()))
    selected = queues["left"][:per_sign] + queues["right"][:per_sign]
    selected_set = set(selected)
    for index in base.ranking:
        if len(selected) >= base.selected_count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    rejected = [
        index for index in range(len(summaries)) if index not in selected_set
    ]
    return SelectionResult(
        selected_indices=tuple(selected),
        valid_indices=base.valid_indices,
        rejected_indices=tuple(rejected),
        alpha=base.alpha,
        selected_count=len(selected),
        ranking=base.ranking,
    )
