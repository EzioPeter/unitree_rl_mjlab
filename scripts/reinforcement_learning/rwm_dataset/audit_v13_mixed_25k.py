"""Independent fail-closed audit for V13 mixed 25K simulator datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_MIX = {
    "expert": 0.45,
    "noisy_expert": 0.25,
    "medium": 0.10,
    "failure_border": 0.15,
    "random": 0.05,
}
COLLECTOR_IDS = {
    "random": 0,
    "expert": 1,
    "noisy_expert": 2,
    "medium": 3,
    "failure_border": 4,
}
SNAPSHOT_WIDTHS = {
    "sim_root_states_local": 13,
    "sim_joint_positions": 12,
    "sim_joint_velocities": 12,
    "sim_action_histories": 12,
    "sim_prev_action_histories": 12,
    "sim_prev_prev_action_histories": 12,
    "sim_snapshot_commands": 3,
}
COMMAND_MODE_WEIGHTS = {
    "stand": 0.08,
    "pure_x": 0.25,
    "pure_y": 0.10,
    "pure_yaw": 0.08,
    "xy": 0.14,
    "x_yaw": 0.17,
    "y_yaw": 0.05,
    "xy_yaw": 0.13,
}
COMMAND_MODE_AXES = {
    "stand": (),
    "pure_x": (0,),
    "pure_y": (1,),
    "pure_yaw": (2,),
    "xy": (0, 1),
    "x_yaw": (0, 2),
    "y_yaw": (1, 2),
    "xy_yaw": (0, 1, 2),
}
COMMAND_ABS_RANGES = {
    "x": (0.05, 0.50),
    "y": (0.03, 0.20),
    "yaw": (0.05, 0.40),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition_id", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing non-empty dataset key {key!r}.")
    return torch.stack(value, dim=0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _shape_and_finite(
    dataset: dict[str, Any],
    key: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    tensor = _stack(dataset, key)
    _require(
        tuple(tensor.shape) == expected_shape,
        f"{key} shape {tuple(tensor.shape)} != {expected_shape}",
    )
    if tensor.is_floating_point():
        _require(bool(torch.isfinite(tensor).all()), f"{key} contains NaN/Inf.")
    return tensor


def _command_distribution(commands: torch.Tensor) -> dict[str, Any]:
    flat = commands.reshape(-1, 3)
    active = flat.abs() > 1.0e-7
    counts: dict[str, int] = {}
    for name, axes in COMMAND_MODE_AXES.items():
        expected = torch.zeros(3, dtype=torch.bool)
        if axes:
            expected[list(axes)] = True
        counts[name] = int((active == expected).all(dim=-1).sum().item())
    total = int(flat.shape[0])
    _require(sum(counts.values()) == total, "Command mode classification is incomplete.")

    axis_names = ("x", "y", "yaw")
    observed_abs_ranges: dict[str, list[float] | None] = {}
    sign_counts: dict[str, dict[str, int]] = {}
    for axis, axis_name in enumerate(axis_names):
        values = flat[:, axis]
        nonzero = values.abs() > 1.0e-7
        active_values = values[nonzero]
        _require(active_values.numel() > 0, f"No non-zero {axis_name} commands found.")
        observed_abs_ranges[axis_name] = [
            float(active_values.abs().min().item()),
            float(active_values.abs().max().item()),
        ]
        sign_counts[axis_name] = {
            "negative": int((active_values < 0).sum().item()),
            "positive": int((active_values > 0).sum().item()),
        }
        lo, hi = COMMAND_ABS_RANGES[axis_name]
        _require(
            bool((active_values.abs() >= lo - 1.0e-6).all()),
            f"{axis_name} command magnitude is below configured range.",
        )
        _require(
            bool((active_values.abs() <= hi + 1.0e-6).all()),
            f"{axis_name} command magnitude is above configured range.",
        )

    return {
        "counts": counts,
        "ratios": {name: count / total for name, count in counts.items()},
        "sign_counts": sign_counts,
        "observed_abs_ranges": observed_abs_ranges,
    }


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    report_path = Path(args.selection_report).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    _require(dataset_path.is_file(), f"Dataset does not exist: {dataset_path}")
    _require(report_path.is_file(), f"Selection report does not exist: {report_path}")
    _require(not output_path.exists(), f"Refusing to overwrite audit report: {output_path}")

    dataset_sha256 = _sha256(dataset_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("schema") == "go2_v13_mixed_stratified_selection_v1",
        f"Unexpected selection report schema: {report.get('schema')!r}",
    )
    _require(
        str(report.get("condition_id")) == str(args.condition_id),
        "Selection report condition does not match.",
    )
    _require(
        str(report.get("output_sha256")) == dataset_sha256,
        "Selection report dataset SHA256 does not match the selected dataset.",
    )
    launch_manifest_path = Path(str(report.get("launch_manifest_path", ""))).expanduser().resolve()
    _require(launch_manifest_path.is_file(), f"Launch manifest does not exist: {launch_manifest_path}")
    launch_manifest_sha256 = _sha256(launch_manifest_path)
    _require(
        str(report.get("launch_manifest_sha256")) == launch_manifest_sha256,
        "Selection report launch manifest SHA256 does not match.",
    )

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    metadata = dict(dataset.get("metadata") or {})
    _require(
        str(metadata.get("condition_id")) == str(args.condition_id),
        "Dataset metadata condition_id does not match.",
    )
    _require(
        metadata.get("subset_kind") == "v13_mixed_environment_columns",
        f"Unexpected subset_kind: {metadata.get('subset_kind')!r}",
    )
    _require(int(dataset.get("num_envs", -1)) == 25, "Selected dataset num_envs must be 25.")
    _require(int(dataset.get("capacity", -1)) == 25_000, "Selected capacity must be 25K.")
    _require(int(metadata.get("num_time_steps", -1)) == 1000, "num_time_steps must be 1000.")
    _require(int(metadata.get("num_transitions", -1)) == 25_000, "num_transitions must be 25K.")
    _require(bool(metadata.get("save_trace_snapshots")), "Source did not enable trace snapshots.")
    _require(bool(metadata.get("trace_snapshots_required")), "Selection did not require snapshots.")
    _require(
        str(metadata.get("launch_manifest_sha256")) == launch_manifest_sha256,
        "Dataset metadata launch manifest SHA256 does not match.",
    )

    prefix = (1000, 25)
    states = _shape_and_finite(dataset, "states", (*prefix, 45))
    next_states = _shape_and_finite(dataset, "next_states", (*prefix, 45))
    _shape_and_finite(dataset, "actions", (*prefix, 12))
    observations = _shape_and_finite(dataset, "observations", (*prefix, 45))
    next_observations = _shape_and_finite(dataset, "next_observations", (*prefix, 45))
    commands = _shape_and_finite(dataset, "commands", (*prefix, 3))
    prev_actions = _shape_and_finite(dataset, "prev_actions", (*prefix, 12))
    _shape_and_finite(dataset, "contacts", (*prefix, int(dataset["contact_dim"])))
    _shape_and_finite(
        dataset,
        "terminations",
        (*prefix, int(dataset["termination_dim"])),
    )
    _require(bool(torch.isfinite(next_states[..., :3]).all()), "BLV targets contain NaN/Inf.")

    for key, width in SNAPSHOT_WIDTHS.items():
        _shape_and_finite(dataset, key, (*prefix, width))
    trace_valid = _stack(dataset, "trace_valid_masks").bool()
    _require(tuple(trace_valid.shape[:2]) == prefix, "trace_valid_masks prefix mismatch.")
    _require(bool(trace_valid.all()), "Not every selected row is valid for TRACE reset.")

    collector_types = _stack(dataset, "collector_types").long()
    _require(tuple(collector_types.shape) == prefix, "collector_types shape mismatch.")
    unique_collector_ids = set(int(value) for value in torch.unique(collector_types).tolist())
    _require(
        unique_collector_ids.issubset(set(COLLECTOR_IDS.values())),
        f"Unknown collector IDs: {sorted(unique_collector_ids)}",
    )
    counts_by_id = torch.bincount(
        collector_types.reshape(-1),
        minlength=max(COLLECTOR_IDS.values()) + 1,
    )
    collector_counts = {
        name: int(counts_by_id[collector_id])
        for name, collector_id in COLLECTOR_IDS.items()
    }
    collector_ratios = {
        name: count / 25_000 for name, count in collector_counts.items()
    }
    collector_ratio_errors = {
        name: collector_ratios[name] - EXPECTED_MIX[name] for name in EXPECTED_MIX
    }
    max_ratio_error = max(abs(value) for value in collector_ratio_errors.values())
    missing_collectors = [
        name for name, count in collector_counts.items() if count <= 0
    ]
    _require(
        not missing_collectors,
        f"Selected V13 mixed dataset is missing collectors: {missing_collectors}.",
    )

    reconstructed_full_policy_obs = torch.cat(
        (
            states[..., 0:9],
            commands,
            states[..., 9:33],
            prev_actions,
        ),
        dim=-1,
    )
    _require(
        tuple(reconstructed_full_policy_obs.shape) == (*prefix, 48),
        "Full-state policy observation reconstruction must be 48D.",
    )
    legacy_collector_observation_max_error = float(
        torch.max(torch.abs(reconstructed_full_policy_obs[..., 3:] - observations)).item()
    )
    _require(
        legacy_collector_observation_max_error <= 1.0e-6,
        "Stored 45D collector observation is not the [3:48] projection of the "
        f"reconstructed full observation: {legacy_collector_observation_max_error}.",
    )
    reconstructed_next_full_policy_obs = torch.cat(
        (next_states[..., :3], next_observations),
        dim=-1,
    )
    _require(
        tuple(reconstructed_next_full_policy_obs.shape) == (*prefix, 48),
        "Next full-state policy observation reconstruction must be 48D.",
    )
    _require(
        bool(torch.isfinite(reconstructed_next_full_policy_obs).all()),
        "Reconstructed next full-state policy observations contain NaN/Inf.",
    )

    selected_command_distribution = _command_distribution(commands)
    raw_command_counts = metadata.get("command_mode_counts")
    _require(isinstance(raw_command_counts, dict), "Raw command_mode_counts metadata is missing.")
    normalized_raw_command_counts = {
        name: int(raw_command_counts.get(name, 0)) for name in COMMAND_MODE_WEIGHTS
    }
    raw_command_total = sum(normalized_raw_command_counts.values())
    _require(raw_command_total == 1_024_000, "Raw command counts must sum to 1,024,000.")
    _require(
        all(count > 0 for count in normalized_raw_command_counts.values()),
        "Raw source pool is missing at least one command mode.",
    )

    episode_ids = _stack(dataset, "episode_ids").long()
    timesteps = _stack(dataset, "timesteps").long()
    dones = _stack(dataset, "dones").bool()
    timeouts = _stack(dataset, "timeouts").bool()
    _require(tuple(episode_ids.shape) == prefix, "episode_ids shape mismatch.")
    _require(tuple(timesteps.shape) == prefix, "timesteps shape mismatch.")
    _require(tuple(dones.shape) == prefix, "dones shape mismatch.")
    _require(tuple(timeouts.shape) == prefix, "timeouts shape mismatch.")
    episode_changes = episode_ids[1:] != episode_ids[:-1]
    unchanged = ~episode_changes
    timestep_increment_errors = int(
        ((timesteps[1:] != timesteps[:-1] + 1) & unchanged).sum().item()
    )
    _require(
        timestep_increment_errors == 0,
        f"Found {timestep_increment_errors} timestep increments inconsistent with episode_ids.",
    )

    source_env_ids = _stack(dataset, "source_env_ids").long()
    _require(tuple(source_env_ids.shape) == prefix, "source_env_ids shape mismatch.")
    _require(
        bool((source_env_ids == source_env_ids[0:1]).all()),
        "source_env_ids must remain constant down each selected column.",
    )
    _require(
        int(torch.unique(source_env_ids[0]).numel()) == 25,
        "Selected source environment IDs must be unique.",
    )

    audit = {
        "schema": "go2_v13_mixed_25k_fullstate_audit_v2",
        "status": "passed",
        "condition_id": str(args.condition_id),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "selection_report_path": str(report_path),
        "selection_report_sha256": _sha256(report_path),
        "launch_manifest_path": str(launch_manifest_path),
        "launch_manifest_sha256": launch_manifest_sha256,
        "shape": {
            "time_steps": 1000,
            "num_envs": 25,
            "transitions": 25_000,
            "state_dim": 45,
            "stored_collector_observation_dim": 45,
            "reconstructed_policy_observation_dim": 48,
            "action_dim": 12,
        },
        "requested_collector_mix": EXPECTED_MIX,
        "actual_collector_counts": collector_counts,
        "actual_collector_ratios": collector_ratios,
        "collector_ratio_errors": collector_ratio_errors,
        "max_abs_collector_ratio_error": max_ratio_error,
        "collector_ratio_acceptance": (
            "reported_not_bounded; reference resamples collectors at command/episode "
            "boundaries, so transition occupancy is condition/survival dependent"
        ),
        "requested_command_sampling_weights": COMMAND_MODE_WEIGHTS,
        "command_sampling_semantics": (
            "random weighted sampling at initialization, 120-300-step expiry, "
            "and episode reset; realized transition occupancy is reported_not_bounded"
        ),
        "raw_source_command_counts": normalized_raw_command_counts,
        "raw_source_command_ratios": {
            name: count / raw_command_total
            for name, count in normalized_raw_command_counts.items()
        },
        "selected_command_counts": selected_command_distribution["counts"],
        "selected_command_ratios": selected_command_distribution["ratios"],
        "selected_command_sign_counts": selected_command_distribution["sign_counts"],
        "selected_command_observed_abs_ranges": selected_command_distribution[
            "observed_abs_ranges"
        ],
        "command_ratio_acceptance": "reported_not_bounded_no_hard_quota",
        "full_policy_observation_reconstructable": True,
        "legacy_collector_observation_max_error": legacy_collector_observation_max_error,
        "actor_reconstruction_max_error": legacy_collector_observation_max_error,
        "unique_episode_ids": int(torch.unique(episode_ids).numel()),
        "episode_changes": int(episode_changes.sum().item()),
        "done_count": int(dones.sum().item()),
        "timeout_count": int(timeouts.sum().item()),
        "all_trace_valid": True,
        "snapshot_widths": SNAPSHOT_WIDTHS,
        "base_lin_vel_target_finite": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[V13-MixedAudit] PASS condition={args.condition_id}")
    print(f"[V13-MixedAudit] dataset_sha256={dataset_sha256}")
    print(
        "[V13-MixedAudit] collector_ratios="
        f"{json.dumps(collector_ratios, sort_keys=True)}"
    )
    print(
        "[V13-MixedAudit] selected_command_ratios="
        f"{json.dumps(selected_command_distribution['ratios'], sort_keys=True)}"
    )
    print(
        "[V13-MixedAudit] legacy_collector_observation_max_error="
        f"{legacy_collector_observation_max_error:.9g}"
    )


if __name__ == "__main__":
    main()
