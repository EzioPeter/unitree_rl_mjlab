"""Compact numeric context describing value to the current policy."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .feedback_pairs import command_region, strip_dataset_command_mode_metadata
from .schemas import (
    COMMAND_REGION_FEATURE_NAMES,
    COMMAND_REGIONS,
    POLICY_CONTEXT_FEATURE_NAMES,
    SUMMARY_SCHEMA_HASH,
)


def policy_cohort(
    summary: Mapping[str, Any],
    planar_command_scales: Sequence[float],
) -> str:
    return command_region(summary, planar_command_scales)


def _finite(summary: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(summary.get(name, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _trajectory_gap(summary: Mapping[str, Any]) -> float:
    tracking = [
        value
        for axis in ("vx", "vy", "yaw")
        if bool((summary.get("command_active") or {}).get(axis, False))
        and (
            value := _finite(
                summary, f"tracking_{axis}_normalized_mae_full"
            )
        )
        is not None
    ]
    tracking_score = (
        float(np.mean([1.0 - np.exp(-max(0.0, value)) for value in tracking]))
        if tracking
        else 0.0
    )
    direction = _finite(summary, "command_direction_violation_rate")
    response = _finite(summary, "minimum_active_axis_response_full")
    components = [tracking_score]
    if direction is not None:
        components.append(float(np.clip(direction, 0.0, 1.0)))
    if response is not None:
        components.append(1.0 - float(np.clip(response, 0.0, 1.0)))
    return float(np.clip(np.mean(components), 0.0, 1.0))


def attach_policy_context(
    summaries: Sequence[Mapping[str, Any]],
    *,
    planar_command_scales: Sequence[float],
    replay_cohort_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Attach two fixed numeric MLP features and compact LLM support metadata."""

    if not summaries:
        return []
    cohorts = [
        policy_cohort(summary, planar_command_scales) for summary in summaries
    ]
    candidate_counts = Counter(cohorts)
    gaps: dict[str, list[float]] = defaultdict(list)
    for cohort, summary in zip(cohorts, summaries, strict=True):
        if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            raise ValueError("Policy context received a stale trajectory summary.")
        gaps[cohort].append(_trajectory_gap(summary))
    global_gap = float(
        np.median([value for values in gaps.values() for value in values])
    )
    cohort_gap: dict[str, float] = {}
    for cohort, values in gaps.items():
        support = len(values)
        reliability = support / (support + 20.0)
        cohort_gap[cohort] = float(
            np.clip(
                reliability * float(np.median(values))
                + (1.0 - reliability) * global_gap,
                0.0,
                1.0,
            )
        )

    replay_total = sum(max(0, int(value)) for value in replay_cohort_counts.values())
    candidate_total = len(summaries)
    enriched: list[dict[str, Any]] = []
    for summary, cohort in zip(summaries, cohorts, strict=True):
        candidate_share = candidate_counts[cohort] / candidate_total
        if replay_total == 0:
            shortage = 1.0
        else:
            replay_share = max(0, int(replay_cohort_counts.get(cohort, 0))) / replay_total
            shortage = 1.0 - min(1.0, replay_share / max(candidate_share, 1.0e-12))
        row = strip_dataset_command_mode_metadata(summary)
        for region, feature_name in zip(
            COMMAND_REGIONS, COMMAND_REGION_FEATURE_NAMES, strict=True
        ):
            row[feature_name] = float(cohort == region)
        row["policy_gap_score"] = cohort_gap[cohort]
        row["replay_shortage_score"] = float(np.clip(shortage, 0.0, 1.0))
        row["policy_context"] = {
            "cohort": cohort,
            "candidate_count": int(candidate_counts[cohort]),
            "replay_selected_count": int(replay_cohort_counts.get(cohort, 0)),
            "policy_gap_score": row["policy_gap_score"],
            "replay_shortage_score": row["replay_shortage_score"],
        }
        missing = set(map(str, row.get("missing_features", ())))
        missing.difference_update(POLICY_CONTEXT_FEATURE_NAMES)
        row["missing_features"] = sorted(missing)
        enriched.append(row)
    return enriched
