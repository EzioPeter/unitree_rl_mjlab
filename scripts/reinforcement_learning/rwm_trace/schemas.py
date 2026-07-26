"""Versioned schemas shared by the final Go2 TRACE Stage-B path."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


SUMMARY_SCHEMA_VERSION = "go2_trace_trajectory_summary_v4"
LLM_DISPLAY_SCHEMA_VERSION = "go2_trace_llm_display_schema_v1"
SCORER_FEATURE_SCHEMA_VERSION = "go2_trace_scorer_feature_schema_v1"
REPLAY_SCHEMA_VERSION = "go2_trace_mutable_replay_v1"
PROMPT_VERSION = "go2_trace_feedback_prompt_v4"
PAIR_SCHEMA_VERSION = "go2_trace_feedback_pair_v2"
LABEL_SCHEMA_VERSION = "go2_trace_feedback_label_v2"

AXES = ("vx", "vy", "yaw")
FEET = ("fr", "fl", "rr", "rl")
COMMAND_MODES = (
    "stand",
    "pure_x",
    "pure_y",
    "pure_yaw",
    "xy",
    "x_yaw",
    "y_yaw",
    "xy_yaw",
)
FORMAL_SOURCE_KINDS = (
    "offline_window",
    "same_start_candidate",
    "cross_start_candidate",
)
COMMAND_REGIONS = ("front", "back", "left", "right", "pure_yaw", "stand")
COMMAND_REGION_DEFINITION = {
    "planar_coordinates": (
        "x=mean(command_vx)/planar_command_scales[0]",
        "y=mean(command_vy)/planar_command_scales[1]",
    ),
    "boundaries": (
        "front:x>=abs(y)",
        "back:x<=-abs(y)",
        "left:y>abs(x)",
        "right:otherwise",
    ),
    "pure_yaw_direct": True,
    "stand_direct": True,
    "diagonal_ties": ("front", "back"),
    "cross_region_pair_fraction": 0.2,
    "balance_regions": False,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _feature_names() -> tuple[str, ...]:
    names = [
        "simulator_return",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "reward_trend_slope",
        "survival_length",
        "survival_fraction",
        "terminal_flag",
        "done_step",
        "expected_trajectory_length",
    ]
    names.extend(f"command_mode_{mode}" for mode in COMMAND_MODES)
    for axis in AXES:
        names.extend(
            (
                f"command_{axis}_abs_mean",
                f"{axis}_active",
                f"actual_{axis}_mean",
                f"actual_{axis}_std",
                f"tracking_{axis}_mae_full",
                f"tracking_{axis}_rmse_full",
                f"tracking_{axis}_normalized_mae_full",
                f"tracking_{axis}_bias_full",
                f"tracking_{axis}_mae_steady",
                f"tracking_{axis}_rmse_steady",
                f"tracking_{axis}_normalized_mae_steady",
                f"tracking_{axis}_bias_steady",
                f"realization_{axis}_full",
                f"realization_{axis}_steady",
                f"direction_correct_{axis}",
                f"direction_violation_{axis}",
                f"overshoot_fraction_{axis}",
                f"overshoot_magnitude_{axis}",
                f"undershoot_fraction_{axis}",
                f"undershoot_magnitude_{axis}",
                f"drift_abs_mean_{axis}",
                f"drift_abs_max_{axis}",
                f"drift_abs_integral_{axis}",
            )
        )
    names.extend(
        (
            "minimum_active_axis_response_full",
            "minimum_active_axis_response_steady",
            "command_direction_correct_fraction",
            "command_direction_violation_rate",
            "command_projected_displacement",
            "expected_command_displacement",
            "command_displacement_realization_ratio",
            "cross_track_velocity_abs_mean",
            "cross_track_integral",
            "yaw_displacement",
            "expected_yaw_displacement",
            "yaw_displacement_realization_ratio",
            "tilt_mean",
            "tilt_max",
            "tilt_final",
            "tilt_trend_slope",
            "roll_pitch_rate_rms",
            "roll_pitch_rate_max",
            "base_vertical_velocity_rms",
            "base_vertical_velocity_max_abs",
            "base_height_mean",
            "base_height_min",
            "base_height_final",
            "base_height_drift",
            "base_height_trend_slope",
            "action_norm_mean",
            "action_norm_std",
            "action_norm_max",
            "action_delta_norm_mean",
            "action_delta_norm_std",
            "action_delta_norm_max",
            "action_saturation_fraction",
            "state_delta_norm_mean",
            "state_delta_norm_max",
            "joint_velocity_rms",
            "joint_velocity_max_abs",
            "actuator_force_rms",
            "actuator_force_max_abs",
        )
    )
    for foot in FEET:
        names.extend(
            (
                f"contact_fraction_{foot}",
                f"contact_switch_count_{foot}",
                f"longest_stance_steps_{foot}",
                f"longest_swing_steps_{foot}",
                f"foot_swing_count_{foot}",
                f"foot_height_mean_{foot}",
                f"foot_height_peak_{foot}",
                f"foot_speed_mean_{foot}",
            )
        )
    if len(names) != len(set(names)):
        raise RuntimeError("Scorer feature schema contains duplicate names.")
    return tuple(names)


SCORER_FEATURE_NAMES = _feature_names()
SCORER_EXPANDED_FEATURE_NAMES = SCORER_FEATURE_NAMES + tuple(
    f"{name}_missing" for name in SCORER_FEATURE_NAMES
)
SCORER_FEATURE_SCHEMA_HASH = canonical_sha256(
    {
        "version": SCORER_FEATURE_SCHEMA_VERSION,
        "ordered_names": SCORER_FEATURE_NAMES,
        "missing_mask_suffix": True,
    }
)
SUMMARY_SCHEMA_HASH = canonical_sha256(
    {
        "version": SUMMARY_SCHEMA_VERSION,
        "axes": AXES,
        "feet": FEET,
        "command_modes": COMMAND_MODES,
        "pairing_metadata": ("command_mean.vx", "command_mean.vy", "command_mean.yaw"),
    }
)
LLM_DISPLAY_SCHEMA_HASH = canonical_sha256(
    {
        "version": LLM_DISPLAY_SCHEMA_VERSION,
        "summary_schema_hash": SUMMARY_SCHEMA_HASH,
        "required_numeric_features": SCORER_FEATURE_NAMES,
        "source_kinds": FORMAL_SOURCE_KINDS,
    }
)
PROMPT_HASH = canonical_sha256(
    {
        "version": PROMPT_VERSION,
        "display_schema_hash": LLM_DISPLAY_SCHEMA_HASH,
        "source_kinds": FORMAL_SOURCE_KINDS,
        "command_regions": COMMAND_REGIONS,
        "command_region_definition": COMMAND_REGION_DEFINITION,
        "priority": (
            "posture_survival_veto",
            "marginal_coverage_and_corrective_value_primary",
            "informative_imperfection_over_trivial_success",
            "task_difficulty_distinct_from_catastrophic_failure",
            "control_contact_learnability",
            "return_visible_diagnostic",
        ),
        "anti_shortcuts": (
            "do_not_prefer_low_error_alone",
            "do_not_prefer_easy_common_stand_front_back",
            "do_not_prefer_catastrophic_failure",
            "value_viable_lateral_yaw_combined_axis_gaps",
        ),
        "label_fields": (
            "pair_id",
            "feedback",
            "confidence",
            "reason",
            "used_criteria",
            "return_used_as_primary",
            "motion_quality_overrode_return",
            "velocity_tracking_used_as_primary",
            "posture_veto_applied",
        ),
    }
)
PAIR_SCHEMA_HASH = canonical_sha256(
    {
        "version": PAIR_SCHEMA_VERSION,
        "command_regions": COMMAND_REGIONS,
        "command_region_definition": COMMAND_REGION_DEFINITION,
        "comparison_types": (
            "within_command_region",
            "cross_command_region",
        ),
    }
)
REPLAY_SCHEMA_HASH = canonical_sha256(
    {
        "version": REPLAY_SCHEMA_VERSION,
        "sample_keys": (
            "observation",
            "action",
            "reward",
            "terminated",
            "truncated",
            "next_observation",
        ),
        "score_is_metadata_only": True,
    }
)


def schema_manifest() -> dict[str, Any]:
    return {
        "summary": {
            "version": SUMMARY_SCHEMA_VERSION,
            "sha256": SUMMARY_SCHEMA_HASH,
        },
        "llm_display": {
            "version": LLM_DISPLAY_SCHEMA_VERSION,
            "sha256": LLM_DISPLAY_SCHEMA_HASH,
        },
        "scorer_feature": {
            "version": SCORER_FEATURE_SCHEMA_VERSION,
            "sha256": SCORER_FEATURE_SCHEMA_HASH,
            "ordered_names": list(SCORER_FEATURE_NAMES),
            "expanded_ordered_names": list(SCORER_EXPANDED_FEATURE_NAMES),
        },
        "prompt": {"version": PROMPT_VERSION, "sha256": PROMPT_HASH},
        "pair": {"version": PAIR_SCHEMA_VERSION, "sha256": PAIR_SCHEMA_HASH},
        "label": {"version": LABEL_SCHEMA_VERSION},
        "replay": {
            "version": REPLAY_SCHEMA_VERSION,
            "sha256": REPLAY_SCHEMA_HASH,
        },
    }


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise ValueError(
            f"{context} field mismatch: missing={missing}, extra={extra}."
        )
