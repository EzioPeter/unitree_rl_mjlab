"""Deterministic fixed-quota command-region pairs for Go2 TRACE labels."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .go2_feedback_prompt import build_go2_feedback_prompt
from .schemas import (
    COMMAND_REGIONS as SCHEMA_COMMAND_REGIONS,
    PAIR_SCHEMA_HASH,
    PAIR_SCHEMA_VERSION,
    SUMMARY_SCHEMA_HASH,
)


COMMAND_REGIONS = SCHEMA_COMMAND_REGIONS


@dataclass(frozen=True)
class PairQuota:
    within_region: int
    cross_region: int

    def __post_init__(self) -> None:
        if self.within_region < 0 or self.cross_region < 0:
            raise ValueError("Pair quotas must be non-negative.")
        if self.within_region + self.cross_region < 1:
            raise ValueError("At least one feedback pair is required.")


@dataclass(frozen=True)
class _PairBlock:
    region_i: str
    region_j: str
    indices_i: tuple[int, ...]
    indices_j: tuple[int, ...]
    within_region: bool
    count: int


def _validated_planar_scales(
    planar_command_scales: Sequence[float],
) -> tuple[float, float]:
    scales = tuple(map(float, planar_command_scales))
    if (
        len(scales) != 2
        or not all(np.isfinite(value) and value > 0.0 for value in scales)
    ):
        raise ValueError("planar_command_scales must contain two finite positive values.")
    return scales


def command_region(
    summary: Mapping[str, Any],
    planar_command_scales: Sequence[float] = (0.5, 0.2),
    *,
    tolerance: float = 1.0e-6,
) -> str:
    """Classify one summary using normalized signed planar command means."""

    active = summary.get("command_active")
    if not isinstance(active, Mapping):
        raise ValueError(
            "Summary lacks command_active required by pair schema "
            f"{PAIR_SCHEMA_VERSION}; rebuild summaries before pairing."
        )
    vx_active = bool(active.get("vx", False))
    vy_active = bool(active.get("vy", False))
    yaw_active = bool(active.get("yaw", False))
    if not vx_active and not vy_active:
        return "pure_yaw" if yaw_active else "stand"
    scales = _validated_planar_scales(planar_command_scales)
    command_mean = summary.get("command_mean")
    if not isinstance(command_mean, Mapping):
        raise ValueError(
            "Summary lacks signed command_mean required by pair schema "
            f"{PAIR_SCHEMA_VERSION}; rebuild summaries before pairing."
        )
    vx = float(command_mean.get("vx", float("nan")))
    vy = float(command_mean.get("vy", float("nan")))
    if not np.isfinite(vx) or not np.isfinite(vy):
        raise ValueError("Signed planar command means must be finite.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("command-region tolerance must be finite and non-negative.")
    x = vx / scales[0]
    y = vy / scales[1]
    if x >= abs(y) - tolerance:
        return "front"
    if x <= -abs(y) + tolerance:
        return "back"
    if y > abs(x) + tolerance:
        return "left"
    return "right"


def _sample_ranks_without_replacement(
    population_size: int,
    count: int,
    *,
    rng: np.random.Generator,
    kind: str,
) -> list[int]:
    """Floyd sampling in O(count) memory without materializing the population."""

    if population_size < count:
        raise ValueError(
            f"Requested {count} {kind} pairs but only {population_size} "
            "unique pairs are available."
        )
    if count == 0:
        return []
    selected: set[int] = set()
    for upper in range(population_size - count, population_size):
        candidate = int(rng.integers(0, upper + 1))
        selected.add(upper if candidate in selected else candidate)
    ranks = list(selected)
    rng.shuffle(ranks)
    return ranks


def _within_pair_from_rank(indices: tuple[int, ...], rank: int) -> tuple[int, int]:
    size = len(indices)

    def prefix(left: int) -> int:
        return left * (2 * size - left - 1) // 2

    low, high = 0, size - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if prefix(middle) <= rank:
            low = middle
        else:
            high = middle
    right = low + 1 + (rank - prefix(low))
    return indices[low], indices[right]


def _pair_from_rank(
    blocks: Sequence[_PairBlock],
    cumulative_counts: Sequence[int],
    rank: int,
) -> tuple[int, int, _PairBlock]:
    block_index = bisect.bisect_right(cumulative_counts, rank)
    block = blocks[block_index]
    start = 0 if block_index == 0 else cumulative_counts[block_index - 1]
    local_rank = rank - start
    if block.within_region:
        left, right = _within_pair_from_rank(block.indices_i, local_rank)
    else:
        width = len(block.indices_j)
        left = block.indices_i[local_rank // width]
        right = block.indices_j[local_rank % width]
    return left, right, block


def _blocks_by_region(
    summaries: Sequence[Mapping[str, Any]],
    *,
    planar_command_scales: Sequence[float],
) -> tuple[list[_PairBlock], list[_PairBlock], list[str]]:
    regions = [
        command_region(summary, planar_command_scales) for summary in summaries
    ]
    grouped: dict[str, tuple[int, ...]] = {}
    for region in COMMAND_REGIONS:
        grouped[region] = tuple(
            sorted(
                (index for index, value in enumerate(regions) if value == region),
                key=lambda index: str(summaries[index]["trajectory_id"]),
            )
        )
    within: list[_PairBlock] = []
    cross: list[_PairBlock] = []
    for region in COMMAND_REGIONS:
        indices = grouped[region]
        count = len(indices) * (len(indices) - 1) // 2
        if count:
            within.append(
                _PairBlock(region, region, indices, indices, True, count)
            )
    for left_index, region_i in enumerate(COMMAND_REGIONS):
        for region_j in COMMAND_REGIONS[left_index + 1 :]:
            indices_i, indices_j = grouped[region_i], grouped[region_j]
            count = len(indices_i) * len(indices_j)
            if count:
                cross.append(
                    _PairBlock(
                        region_i,
                        region_j,
                        indices_i,
                        indices_j,
                        False,
                        count,
                    )
                )
    return within, cross, regions


def _sample_blocks(
    blocks: Sequence[_PairBlock],
    count: int,
    *,
    rng: np.random.Generator,
    kind: str,
) -> list[tuple[int, int, _PairBlock]]:
    cumulative: list[int] = []
    total = 0
    for block in blocks:
        total += block.count
        cumulative.append(total)
    ranks = _sample_ranks_without_replacement(total, count, rng=rng, kind=kind)
    return [_pair_from_rank(blocks, cumulative, rank) for rank in ranks]


def build_feedback_pairs(
    summaries: Sequence[Mapping[str, Any]],
    *,
    quota: PairQuota,
    pair_prefix: str,
    seed: int,
    planar_command_scales: Sequence[float],
) -> list[dict[str, Any]]:
    """Sample exact global within/cross-region quotas without scorer inputs."""

    if len(summaries) < 2:
        raise ValueError("At least two summaries are needed.")
    scales = _validated_planar_scales(planar_command_scales)
    normalized = [dict(item) for item in summaries]
    for item in normalized:
        if item.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            raise ValueError("Every feedback candidate must use the current summary schema.")
        if not str(item.get("trajectory_id", "")):
            raise ValueError("Every feedback candidate needs a non-empty trajectory_id.")
    within_blocks, cross_blocks, _regions = _blocks_by_region(
        normalized, planar_command_scales=scales
    )
    rng = np.random.default_rng(int(seed))
    selected = [
        *_sample_blocks(
            within_blocks,
            quota.within_region,
            rng=rng,
            kind="within-command-region",
        ),
        *_sample_blocks(
            cross_blocks,
            quota.cross_region,
            rng=rng,
            kind="cross-command-region",
        ),
    ]
    rng.shuffle(selected)
    rows: list[dict[str, Any]] = []
    for index, (left, right, block) in enumerate(selected):
        comparison_type = (
            "within_command_region"
            if block.within_region
            else "cross_command_region"
        )
        row: dict[str, Any] = {
            "pair_schema_version": PAIR_SCHEMA_VERSION,
            "pair_schema_hash": PAIR_SCHEMA_HASH,
            "pair_id": f"{pair_prefix}_{index:06d}",
            "comparison_type": comparison_type,
            "command_region_i": block.region_i,
            "command_region_j": block.region_j,
            "same_command_region": block.within_region,
            "planar_command_scales": list(scales),
            "trajectory_i": normalized[left],
            "trajectory_j": normalized[right],
        }
        row["prompt"] = build_go2_feedback_prompt(row)
        rows.append(row)
    actual_within = sum(row["same_command_region"] for row in rows)
    if (
        actual_within != quota.within_region
        or len(rows) - actual_within != quota.cross_region
    ):
        raise RuntimeError("Pair construction did not preserve the exact region quota.")
    return rows
