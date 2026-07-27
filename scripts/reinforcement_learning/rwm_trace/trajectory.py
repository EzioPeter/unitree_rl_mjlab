"""Fixed-schema Go2 trajectory summaries for the formal TRACE Stage-B path.

The summary is task specific: command velocity tracking is the primary signal,
upright posture is a safety veto, and reward/return remain visible diagnostics.
The learned scorer consumes a fixed ordered numeric view of this summary.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch

from .schemas import (
    AXES,
    COMMAND_MODES,
    FEET,
    FORMAL_SOURCE_KINDS,
    LLM_DISPLAY_SCHEMA_HASH,
    LLM_DISPLAY_SCHEMA_VERSION,
    SCORER_FEATURE_NAMES,
    SUMMARY_SCHEMA_HASH,
    SUMMARY_SCHEMA_VERSION,
)


_MODE_BY_ACTIVE_AXES = {
    (False, False, False): "stand",
    (True, False, False): "pure_x",
    (False, True, False): "pure_y",
    (False, False, True): "pure_yaw",
    (True, True, False): "xy",
    (True, False, True): "x_yaw",
    (False, True, True): "y_yaw",
    (True, True, True): "xy_yaw",
}


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _required_array(
    trajectory: Mapping[str, Any],
    key: str,
    shape_tail: tuple[int, ...],
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    if key not in trajectory:
        raise KeyError(f"Go2 TRACE trajectory is missing required field {key!r}.")
    value = _array(trajectory[key], dtype=dtype)
    if value.ndim != 1 + len(shape_tail) or value.shape[1:] != shape_tail:
        raise ValueError(
            f"Go2 TRACE field {key!r} must have shape [T,{','.join(map(str, shape_tail))}], "
            f"got {value.shape}."
        )
    return value


def _positive_vector(value: Any, name: str, width: int) -> np.ndarray:
    result = _array(value).reshape(-1)
    if result.shape != (width,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain {width} finite positive values, got {value!r}.")
    return result


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}number, got {value!r}.")
    return result


def _mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def _trend(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, values, 1)[0])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return float("nan")
    return float(numerator / denominator)


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for item in np.asarray(mask, dtype=bool).reshape(-1):
        current = current + 1 if bool(item) else 0
        longest = max(longest, current)
    return int(longest)


def _switch_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    return int(np.count_nonzero(mask[1:] != mask[:-1])) if len(mask) > 1 else 0


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = (quaternion[..., index] for index in range(4))
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _identity(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    source_kind = str(trajectory.get("source_kind", "same_start_candidate"))
    if source_kind not in FORMAL_SOURCE_KINDS:
        raise ValueError(
            f"source_kind must be one of {FORMAL_SOURCE_KINDS}, got {source_kind!r}."
        )
    return {
        "trajectory_id": str(trajectory.get("trajectory_id", "")),
        "start_state_id": str(trajectory.get("start_state_id", "")),
        "start_state_key": str(
            trajectory.get("start_state_key", trajectory.get("start_state_id", ""))
        ),
        "comparison_group_key": str(
            trajectory.get(
                "comparison_group_key",
                trajectory.get("start_state_key", trajectory.get("start_state_id", "")),
            )
        ),
        "source_kind": source_kind,
        "dataset_id": str(trajectory.get("dataset_id", "")),
        "task_id": str(trajectory.get("task_id", "")),
        "condition_id": str(trajectory.get("condition_id", "")),
        "candidate_seed": int(trajectory.get("candidate_seed", -1)),
        "realized_snapshot_hash": str(
            trajectory.get("realized_snapshot_hash", "")
        ),
        "simulator_config_hash": str(
            trajectory.get("simulator_config_hash", "")
        ),
        "command_sequence_hash": str(
            trajectory.get("command_sequence_hash", "")
        ),
    }


def summarize_go2_trajectory(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Create the fixed Go2 V3 summary from one transition trajectory.

    Fixed state layout:
    ``[base_lin_vel_b(3), base_ang_vel_b(3), projected_gravity_b(3),
    joint_pos_rel(12), joint_vel(12), actuator_force(12)]``.

    ``next_states[:, (0, 1, 5)]`` is compared with the command attached to the
    same transition. Thresholds and normalization floors are required inputs so
    dataset/task configuration, rather than this function, owns their values.
    """

    states = _required_array(trajectory, "states", (45,))
    next_states = _required_array(trajectory, "next_states", (45,))
    actions = _required_array(trajectory, "actions", (12,))
    commands = _required_array(trajectory, "commands", (3,))
    contacts = _required_array(trajectory, "contacts", (4,))
    rewards = _array(trajectory["rewards"]).reshape(-1) if "rewards" in trajectory else None
    terminations = (
        _array(trajectory["terminations"], dtype=bool).reshape(-1)
        if "terminations" in trajectory
        else None
    )
    if rewards is None or terminations is None:
        raise KeyError("Go2 TRACE summaries require both rewards and terminations.")

    length = int(states.shape[0])
    if length < 1:
        raise ValueError("Go2 TRACE trajectory must contain at least one transition.")
    arrays = (next_states, actions, commands, contacts, rewards, terminations)
    if any(len(value) != length for value in arrays):
        raise ValueError("All Go2 TRACE transition arrays must share the same leading length.")
    if next_states.shape != states.shape:
        raise ValueError("states and next_states must both be [T,45].")
    if not all(np.isfinite(value).all() for value in (states, next_states, actions, commands, contacts, rewards)):
        raise ValueError("Go2 TRACE summary inputs contain non-finite numeric values.")

    step_dt = _finite_scalar(trajectory["step_dt"], "step_dt", positive=True)
    active_thresholds = _positive_vector(
        trajectory["command_active_thresholds"], "command_active_thresholds", 3
    )
    normalization_floors = _positive_vector(
        trajectory["command_normalization_floors"], "command_normalization_floors", 3
    )
    action_saturation_threshold = _finite_scalar(
        trajectory["action_saturation_threshold"],
        "action_saturation_threshold",
        positive=True,
    )
    expected_length = int(trajectory.get("expected_trajectory_length", length))
    if expected_length < length or expected_length < 1:
        raise ValueError("expected_trajectory_length must be at least the realized length.")
    steady_start = min(length - 1, int(trajectory.get("steady_start_step", length // 2)))
    if steady_start < 0:
        raise ValueError("steady_start_step must be non-negative.")
    if not trajectory.get("command_sequence_hash"):
        command_sequence_hash = hashlib.sha256(
            commands.astype(np.float32).tobytes()
        ).hexdigest()
    else:
        command_sequence_hash = str(trajectory["command_sequence_hash"])

    features: dict[str, float] = {
        name: float("nan") for name in SCORER_FEATURE_NAMES
    }
    terminal_indices = np.flatnonzero(terminations)
    features.update(
        {
            "simulator_return": float(np.sum(rewards)),
            "reward_mean": _mean(rewards),
            "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_trend_slope": _trend(rewards),
            "survival_length": float(length),
            "survival_fraction": float(length / expected_length),
            "terminal_flag": float(bool(terminal_indices.size)),
            "done_step": float(terminal_indices[0]) if terminal_indices.size else float(length),
            "expected_trajectory_length": float(expected_length),
        }
    )

    actual = next_states[:, (0, 1, 5)]
    command_mean = np.mean(commands, axis=0)
    command_abs_mean = np.mean(np.abs(commands), axis=0)
    active_axes = tuple(bool(command_abs_mean[i] > active_thresholds[i]) for i in range(3))
    command_mode = _MODE_BY_ACTIVE_AXES[active_axes]
    for mode in COMMAND_MODES:
        features[f"command_mode_{mode}"] = float(mode == command_mode)

    active_full_responses: list[float] = []
    active_steady_responses: list[float] = []
    direction_correct_values: list[float] = []
    direction_violation_values: list[float] = []
    for index, axis in enumerate(AXES):
        cmd = commands[:, index]
        out = actual[:, index]
        error = out - cmd
        steady = slice(steady_start, None)
        active_step = np.abs(cmd) > active_thresholds[index]
        active_step_steady = active_step[steady]
        signed_response = out * np.sign(cmd)
        magnitude_error = signed_response - np.abs(cmd)

        features[f"command_{axis}_abs_mean"] = float(command_abs_mean[index])
        features[f"{axis}_active"] = float(active_axes[index])
        features[f"actual_{axis}_mean"] = _mean(out)
        features[f"actual_{axis}_std"] = float(np.std(out))
        features[f"drift_abs_mean_{axis}"] = _mean(np.abs(out))
        features[f"drift_abs_max_{axis}"] = float(np.max(np.abs(out)))
        features[f"drift_abs_integral_{axis}"] = float(np.sum(np.abs(out)) * step_dt)

        if not active_axes[index]:
            continue
        floor = normalization_floors[index]
        features[f"tracking_{axis}_mae_full"] = _mean(np.abs(error))
        features[f"tracking_{axis}_rmse_full"] = _rms(error)
        features[f"tracking_{axis}_normalized_mae_full"] = _mean(
            np.abs(error) / np.maximum(np.abs(cmd), floor)
        )
        features[f"tracking_{axis}_bias_full"] = _mean(error)
        features[f"tracking_{axis}_mae_steady"] = _mean(np.abs(error[steady]))
        features[f"tracking_{axis}_rmse_steady"] = _rms(error[steady])
        features[f"tracking_{axis}_normalized_mae_steady"] = _mean(
            np.abs(error[steady]) / np.maximum(np.abs(cmd[steady]), floor)
        )
        features[f"tracking_{axis}_bias_steady"] = _mean(error[steady])
        features[f"realization_{axis}_full"] = _safe_ratio(
            _mean(signed_response[active_step]), _mean(np.abs(cmd[active_step]))
        )
        features[f"realization_{axis}_steady"] = (
            _safe_ratio(
                _mean(signed_response[steady][active_step_steady]),
                _mean(np.abs(cmd[steady][active_step_steady])),
            )
            if np.any(active_step_steady)
            else float("nan")
        )
        direction_correct = _mean(signed_response[active_step] > 0.0)
        direction_violation = _mean(signed_response[active_step] < 0.0)
        features[f"direction_correct_{axis}"] = direction_correct
        features[f"direction_violation_{axis}"] = direction_violation
        overshoot = magnitude_error[active_step] > 0.0
        undershoot = magnitude_error[active_step] < 0.0
        features[f"overshoot_fraction_{axis}"] = _mean(overshoot)
        features[f"overshoot_magnitude_{axis}"] = _mean(
            np.maximum(magnitude_error[active_step], 0.0)
        )
        features[f"undershoot_fraction_{axis}"] = _mean(undershoot)
        features[f"undershoot_magnitude_{axis}"] = _mean(
            np.maximum(-magnitude_error[active_step], 0.0)
        )
        active_full_responses.append(features[f"realization_{axis}_full"])
        active_steady_responses.append(features[f"realization_{axis}_steady"])
        direction_correct_values.append(direction_correct)
        direction_violation_values.append(direction_violation)

    features["minimum_active_axis_response_full"] = (
        float(np.nanmin(active_full_responses)) if active_full_responses else float("nan")
    )
    features["minimum_active_axis_response_steady"] = (
        float(np.nanmin(active_steady_responses)) if active_steady_responses else float("nan")
    )
    features["command_direction_correct_fraction"] = (
        _mean(direction_correct_values) if direction_correct_values else float("nan")
    )
    features["command_direction_violation_rate"] = (
        _mean(direction_violation_values) if direction_violation_values else float("nan")
    )

    command_xy = commands[:, :2]
    actual_xy = actual[:, :2]
    command_speed = np.linalg.norm(command_xy, axis=1)
    linear_active_steps = command_speed > float(np.linalg.norm(active_thresholds[:2]))
    unit_command = np.zeros_like(command_xy)
    unit_command[linear_active_steps] = (
        command_xy[linear_active_steps] / command_speed[linear_active_steps, None]
    )
    projected = np.sum(actual_xy * unit_command, axis=1)
    cross_track = np.abs(
        actual_xy[:, 0] * unit_command[:, 1]
        - actual_xy[:, 1] * unit_command[:, 0]
    )
    features["command_projected_displacement"] = float(np.sum(projected) * step_dt)
    features["expected_command_displacement"] = float(np.sum(command_speed) * step_dt)
    features["command_displacement_realization_ratio"] = (
        _safe_ratio(
            features["command_projected_displacement"],
            features["expected_command_displacement"],
        )
        if any(active_axes[:2])
        else float("nan")
    )
    features["cross_track_velocity_abs_mean"] = (
        _mean(cross_track[linear_active_steps])
        if np.any(linear_active_steps)
        else float("nan")
    )
    features["cross_track_integral"] = (
        float(np.sum(cross_track[linear_active_steps]) * step_dt)
        if np.any(linear_active_steps)
        else float("nan")
    )
    features["yaw_displacement"] = float(np.sum(actual[:, 2]) * step_dt)
    features["expected_yaw_displacement"] = float(np.sum(commands[:, 2]) * step_dt)
    features["yaw_displacement_realization_ratio"] = (
        _safe_ratio(features["yaw_displacement"], features["expected_yaw_displacement"])
        if active_axes[2]
        else float("nan")
    )

    gravity = next_states[:, 6:9]
    gravity_norm = np.maximum(np.linalg.norm(gravity, axis=1), 1.0e-12)
    tilt = np.arccos(np.clip(-gravity[:, 2] / gravity_norm, -1.0, 1.0))
    roll_pitch_rate = next_states[:, 3:5]
    vertical_velocity = next_states[:, 2]
    features.update(
        {
            "tilt_mean": _mean(tilt),
            "tilt_max": float(np.max(tilt)),
            "tilt_final": float(tilt[-1]),
            "tilt_trend_slope": _trend(tilt),
            "roll_pitch_rate_rms": _rms(roll_pitch_rate),
            "roll_pitch_rate_max": float(np.max(np.abs(roll_pitch_rate))),
            "base_vertical_velocity_rms": _rms(vertical_velocity),
            "base_vertical_velocity_max_abs": float(np.max(np.abs(vertical_velocity))),
        }
    )

    root_states = trajectory.get("sim_root_states_local")
    if root_states is not None:
        root = _array(root_states)
        if root.shape not in {(length, 13), (length + 1, 13)}:
            raise ValueError("sim_root_states_local must have shape [T,13] or [T+1,13].")
        root = root[-length:]
        if not np.isfinite(root).all():
            raise ValueError("sim_root_states_local contains non-finite values.")
        height = root[:, 2]
        yaw = np.unwrap(_yaw_from_wxyz(root[:, 3:7]))
        features.update(
            {
                "base_height_mean": _mean(height),
                "base_height_min": float(np.min(height)),
                "base_height_final": float(height[-1]),
                "base_height_drift": float(height[-1] - height[0]),
                "base_height_trend_slope": _trend(height),
            }
        )
        root_diagnostics = {
            "available": True,
            "world_xy_displacement": (root[-1, :2] - root[0, :2]).tolist(),
            "world_yaw_change": float(yaw[-1] - yaw[0]),
        }
    else:
        root_diagnostics = {"available": False}

    action_norm = np.linalg.norm(actions, axis=1)
    action_delta_norm = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    state_delta_norm = np.linalg.norm(next_states - states, axis=1)
    features.update(
        {
            "action_norm_mean": _mean(action_norm),
            "action_norm_std": float(np.std(action_norm)),
            "action_norm_max": float(np.max(action_norm)),
            "action_delta_norm_mean": _mean(action_delta_norm) if len(action_delta_norm) else 0.0,
            "action_delta_norm_std": float(np.std(action_delta_norm)) if len(action_delta_norm) else 0.0,
            "action_delta_norm_max": float(np.max(action_delta_norm)) if len(action_delta_norm) else 0.0,
            "action_saturation_fraction": _mean(
                np.any(np.abs(actions) >= action_saturation_threshold, axis=1)
            ),
            "state_delta_norm_mean": _mean(state_delta_norm),
            "state_delta_norm_max": float(np.max(state_delta_norm)),
            "joint_velocity_rms": _rms(next_states[:, 21:33]),
            "joint_velocity_max_abs": float(np.max(np.abs(next_states[:, 21:33]))),
            "actuator_force_rms": _rms(next_states[:, 33:45]),
            "actuator_force_max_abs": float(np.max(np.abs(next_states[:, 33:45]))),
        }
    )

    contact_mask = contacts > 0.5
    foot_positions = trajectory.get("trace_foot_site_positions_w")
    foot_velocities = trajectory.get("trace_foot_site_linear_velocities_w")
    if (foot_positions is None) != (foot_velocities is None):
        raise ValueError("Foot positions and velocities must be supplied together.")
    if foot_positions is not None:
        foot_positions = _array(foot_positions)
        foot_velocities = _array(foot_velocities)
        if foot_positions.shape != (length, 4, 3) or foot_velocities.shape != (length, 4, 3):
            raise ValueError("Foot position/velocity diagnostics must both be [T,4,3].")
    for foot_index, foot in enumerate(FEET):
        mask = contact_mask[:, foot_index]
        features[f"contact_fraction_{foot}"] = _mean(mask)
        features[f"contact_switch_count_{foot}"] = float(_switch_count(mask))
        features[f"longest_stance_steps_{foot}"] = float(_longest_run(mask))
        features[f"longest_swing_steps_{foot}"] = float(_longest_run(~mask))
        features[f"foot_swing_count_{foot}"] = float(
            np.count_nonzero((~mask[1:]) & mask[:-1]) if length > 1 else 0
        )
        if foot_positions is not None:
            features[f"foot_height_mean_{foot}"] = _mean(foot_positions[:, foot_index, 2])
            features[f"foot_height_peak_{foot}"] = float(
                np.max(foot_positions[:, foot_index, 2])
            )
            features[f"foot_speed_mean_{foot}"] = _mean(
                np.linalg.norm(foot_velocities[:, foot_index], axis=1)
            )

    missing = sorted(name for name, value in features.items() if not np.isfinite(value))
    summary: dict[str, Any] = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_schema_hash": SUMMARY_SCHEMA_HASH,
        **_identity(trajectory),
        "step_dt": step_dt,
        "command_mode": command_mode,
        # Signed means are pairing metadata rather than scorer features.  They
        # preserve the commanded planar direction needed by the six-region
        # feedback sampler without changing the trained scorer input schema.
        "command_mean": dict(zip(AXES, map(float, command_mean))),
        "command_active": dict(zip(AXES, active_axes)),
        "command_active_thresholds": dict(zip(AXES, active_thresholds.tolist())),
        "command_normalization_floors": dict(zip(AXES, normalization_floors.tolist())),
        "steady_start_step": steady_start,
        "behavior_return": float(np.sum(rewards)),
        "simulator_return": float(np.sum(rewards)),
        "root_diagnostics": root_diagnostics,
        "foot_kinematics_available": foot_positions is not None,
        "missing_features": missing,
        **features,
    }
    summary["command_sequence_hash"] = command_sequence_hash
    return summary


def build_go2_llm_display(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, strict-JSON-safe view used by the Go2 feedback prompt."""

    if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
        raise ValueError("Cannot display a summary from a different schema.")

    def value(name: str) -> float | None:
        parsed = float(summary.get(name, float("nan")))
        return parsed if np.isfinite(parsed) else None

    axes: dict[str, Any] = {}
    for axis in AXES:
        axes[axis] = {
            "active": bool(summary["command_active"][axis]),
            "command_abs_mean": value(f"command_{axis}_abs_mean"),
            "actual_mean": value(f"actual_{axis}_mean"),
            "tracking_mae": value(f"tracking_{axis}_mae_full"),
            "tracking_normalized_mae": value(f"tracking_{axis}_normalized_mae_full"),
            "tracking_mae_steady": value(f"tracking_{axis}_mae_steady"),
            "realization": value(f"realization_{axis}_full"),
            "realization_steady": value(f"realization_{axis}_steady"),
            "direction_violation_rate": value(f"direction_violation_{axis}"),
            "inactive_axis_drift_mean": (
                value(f"drift_abs_mean_{axis}")
                if not bool(summary["command_active"][axis])
                else None
            ),
        }
    return {
        "display_schema_version": LLM_DISPLAY_SCHEMA_VERSION,
        "display_schema_hash": LLM_DISPLAY_SCHEMA_HASH,
        "trajectory_id": str(summary.get("trajectory_id", "")),
        "source_kind": str(summary.get("source_kind", "")),
        "command_mode": str(summary.get("command_mode", "")),
        "velocity_tracking": axes,
        "progress": {
            name: value(name)
            for name in (
                "command_projected_displacement",
                "expected_command_displacement",
                "command_displacement_realization_ratio",
                "cross_track_velocity_abs_mean",
                "yaw_displacement_realization_ratio",
            )
        },
        "posture_and_survival": {
            name: value(name)
            for name in (
                "survival_fraction",
                "terminal_flag",
                "tilt_mean",
                "tilt_max",
                "tilt_final",
                "roll_pitch_rate_rms",
                "base_vertical_velocity_rms",
                "base_height_min",
                "base_height_drift",
            )
        },
        "control_and_contact": {
            name: value(name)
            for name in (
                "action_norm_mean",
                "action_delta_norm_mean",
                "action_saturation_fraction",
                "joint_velocity_rms",
                "actuator_force_rms",
                *(f"contact_switch_count_{foot}" for foot in FEET),
            )
        },
        "return_diagnostic": {
            "simulator_return": value("simulator_return"),
            "reward_mean": value("reward_mean"),
            "reward_std": value("reward_std"),
        },
        "missing_features": list(summary.get("missing_features", ())),
    }
