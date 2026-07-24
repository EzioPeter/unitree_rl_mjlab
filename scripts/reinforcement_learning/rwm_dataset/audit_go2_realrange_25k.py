"""Fail-closed audit for real-command-range Go2 25K datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


MODE_TARGETS = {
    "stand": 2000,
    "pure_x": 6250,
    "pure_y": 2500,
    "pure_yaw": 2000,
    "xy": 3500,
    "x_yaw": 4250,
    "y_yaw": 1250,
    "xy_yaw": 3250,
}
EXPECTED_EDGES = {
    "x": [0.08, 0.22, 0.36, 0.50],
    "y": [0.05, 0.10, 0.15, 0.20],
    "yaw": [0.08, 0.186667, 0.293333, 0.40],
}
SNAPSHOT_REQUIRED_KEYS = (
    "root_state_local",
    "joint_position",
    "joint_velocity",
    "action",
    "prev_action",
    "prev_prev_action",
    "command",
)
SNAPSHOT_EXPECTED_DIMS = {
    "root_state_local": 13,
    "joint_position": 12,
    "joint_velocity": 12,
    "action": 12,
    "prev_action": 12,
    "prev_prev_action": 12,
    "command": 3,
}
SNAPSHOT_DATASET_KEYS = {
    "root_state_local": "sim_root_states_local",
    "joint_position": "sim_joint_positions",
    "joint_velocity": "sim_joint_velocities",
    "action": "sim_action_histories",
    "prev_action": "sim_prev_action_histories",
    "prev_prev_action": "sim_prev_prev_action_histories",
    "command": "sim_snapshot_commands",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--num-trajectories", type=int, default=25)
    parser.add_argument("--trajectory-length", type=int, default=1000)
    parser.add_argument(
        "--required-selection-report",
        default=None,
        help="Require the audited selection to contain every env ID from this report.",
    )
    parser.add_argument("--mode-relative-error-limit", type=float, default=0.03)
    parser.add_argument(
        "--allow-mid-trajectory-termination",
        action="store_true",
        help=(
            "Audit a failure-aware 1000x25 rectangular dataset. Episode IDs "
            "and timesteps must mark every internal reset, and training "
            "windows must not cross those boundaries."
        ),
    )
    return parser.parse_args()


def stack_time(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.stack(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    report_path = Path(args.selection_report).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    selection = json.loads(report_path.read_text())
    num_trajectories = int(args.num_trajectories)
    trajectory_length = int(args.trajectory_length)
    transitions = num_trajectories * trajectory_length
    target_scale = float(transitions) / 25000.0

    required_fields = (
        "states",
        "actions",
        "next_states",
        "contacts",
        "terminations",
        "commands",
        "episode_ids",
        "timesteps",
        "collector_types",
        "dones",
        "timeouts",
    )
    for field in required_fields:
        require(field in data, f"Missing required field: {field}")
    snapshot_tensors = {}
    for snapshot_key in SNAPSHOT_REQUIRED_KEYS:
        dataset_key = SNAPSHOT_DATASET_KEYS[snapshot_key]
        require(dataset_key in data, f"Missing snapshot dataset field: {dataset_key}")
        value = stack_time(data[dataset_key])
        expected = (
            trajectory_length,
            num_trajectories,
            SNAPSHOT_EXPECTED_DIMS[snapshot_key],
        )
        require(
            tuple(value.shape) == expected,
            f"{dataset_key}: expected {expected}, got {tuple(value.shape)}",
        )
        require(bool(torch.isfinite(value).all()), f"{dataset_key} contains NaN/Inf")
        snapshot_tensors[snapshot_key] = value

    tensors = {field: stack_time(data[field]) for field in required_fields}
    expected_shapes = {
        "states": (trajectory_length, num_trajectories, 45),
        "actions": (trajectory_length, num_trajectories, 12),
        "next_states": (trajectory_length, num_trajectories, 45),
        "contacts": (trajectory_length, num_trajectories, 4),
        "terminations": (trajectory_length, num_trajectories, 1),
        "commands": (trajectory_length, num_trajectories, 3),
        "episode_ids": (trajectory_length, num_trajectories),
        "timesteps": (trajectory_length, num_trajectories),
        "collector_types": (trajectory_length, num_trajectories),
        "dones": (trajectory_length, num_trajectories),
        "timeouts": (trajectory_length, num_trajectories),
    }
    for field, shape in expected_shapes.items():
        require(tuple(tensors[field].shape) == shape, f"{field}: expected {shape}, got {tuple(tensors[field].shape)}")

    for field in ("states", "actions", "next_states", "contacts", "terminations", "commands"):
        require(bool(torch.isfinite(tensors[field]).all()), f"{field} contains NaN/Inf")

    commands = tensors["commands"].float()
    require(float(commands[..., 0].abs().max()) <= 0.500001, "vx exceeds ±0.5")
    require(float(commands[..., 1].abs().max()) <= 0.200001, "vy exceeds ±0.2")
    require(float(commands[..., 2].abs().max()) <= 0.400001, "yaw exceeds ±0.4")
    require(bool((tensors["collector_types"] == 1).all()), "Dataset is not pure expert")

    episode_ids = tensors["episode_ids"]
    timesteps = tensors["timesteps"]
    terminations = tensors["terminations"][..., 0] > 0.5
    episode_change = episode_ids[1:] != episode_ids[:-1]
    continuity_error = (tensors["next_states"][:-1] - tensors["states"][1:]).abs()
    if args.allow_mid_trajectory_termination:
        require(bool(terminations.any()), "Failure-aware dataset has no termination examples")
        require(
            bool(tensors["dones"][:-1][episode_change].all()),
            "An episode change is not preceded by done",
        )
        require(
            bool(terminations[:-1][episode_change].all()),
            "An episode change is not preceded by termination",
        )
        require(
            bool((timesteps[1:][episode_change] == 0).all()),
            "Timestep does not reset to zero after an episode boundary",
        )
        require(
            bool(
                (
                    timesteps[1:][~episode_change]
                    == timesteps[:-1][~episode_change] + 1
                ).all()
            ),
            "Timestep is not consecutive inside an episode",
        )
        same_episode_error = continuity_error[~episode_change]
        require(
            float(same_episode_error.max()) == 0.0,
            "State/next_state continuity is not exact inside an episode",
        )
    else:
        require(
            int(torch.unique(episode_ids).numel()) == num_trajectories,
            f"Dataset does not contain exactly {num_trajectories} episode IDs",
        )
        require(bool((episode_ids == episode_ids[0:1]).all()), "An environment column changes episode ID")
        require(
            bool(
                (
                    timesteps
                    == torch.arange(trajectory_length, dtype=timesteps.dtype)[:, None]
                ).all()
            ),
            f"Timesteps are not 0..{trajectory_length - 1} for every trajectory",
        )
        require(not bool(terminations[:-1].any()), "Dataset contains a mid-trajectory termination")
        require(bool(terminations[-1].all()), "Every trajectory must end with an explicit boundary")
        require(float(continuity_error.max()) == 0.0, "State/next_state continuity is not exact")

    require(selection.get("condition_id") == args.condition_id, "Selection report condition does not match")
    require(
        selection.get("selected_transitions") == transitions,
        f"Selection report transition count is not {transitions}",
    )
    if args.allow_mid_trajectory_termination:
        require(selection.get("failure_aware_selection") is True, "Selection is not marked failure-aware")
        require(selection.get("whole_trajectory_selection") is False, "Failure-aware selection is mislabeled as whole-trajectory")
        require(
            selection.get("unique_episode_ids", 0) >= num_trajectories,
            f"Failure-aware selection has fewer than {num_trajectories} episode IDs",
        )
    else:
        require(
            selection.get("unique_episode_ids") == num_trajectories,
            f"Selection report does not contain {num_trajectories} episode IDs",
        )
        require(selection.get("whole_trajectory_selection") is True, "Selection is not whole-trajectory")
        require(
            selection.get("eligible_uninterrupted_trajectories", 0)
            >= num_trajectories,
            f"Fewer than {num_trajectories} uninterrupted candidates",
        )
    if args.required_selection_report:
        required = json.loads(
            Path(args.required_selection_report).expanduser().resolve().read_text()
        )
        required_ids = set(int(value) for value in required["selected_env_ids"])
        selected_ids = set(int(value) for value in selection["selected_env_ids"])
        require(
            required_ids.issubset(selected_ids),
            "50K selection is not a superset of the required 25K selection",
        )
    for axis, expected in EXPECTED_EDGES.items():
        actual = selection.get("magnitude_bin_edges", {}).get(axis)
        require(actual is not None, f"Missing {axis} magnitude edges")
        require(max(abs(float(a) - float(b)) for a, b in zip(actual, expected, strict=True)) < 1.0e-6, f"{axis} magnitude edges are wrong")

    actual_modes = selection.get("actual_mode_counts", {})
    scaled_mode_targets = {
        mode: target * target_scale for mode, target in MODE_TARGETS.items()
    }
    mode_errors = {
        mode: (int(actual_modes[mode]) - target) / target
        for mode, target in scaled_mode_targets.items()
    }
    require(
        max(abs(value) for value in mode_errors.values()) <= args.mode_relative_error_limit,
        f"Command-mode relative error exceeds {args.mode_relative_error_limit}",
    )

    audit = {
        "status": "passed",
        "condition_id": args.condition_id,
        "dataset": str(dataset_path),
        "shape": [trajectory_length, num_trajectories],
        "transitions": transitions,
        "unique_episode_ids": int(torch.unique(episode_ids).numel()),
        "mid_trajectory_terminations": int(terminations[:-1].sum()),
        "episode_boundaries": int(episode_change.sum()),
        "failure_aware_selection": bool(args.allow_mid_trajectory_termination),
        "finite": True,
        "pure_expert": True,
        "command_abs_max": commands.abs().amax(dim=(0, 1)).tolist(),
        "mode_relative_errors": mode_errors,
        "state_continuity_max_error": float(
            continuity_error[~episode_change].max()
            if args.allow_mid_trajectory_termination
            else continuity_error.max()
        ),
        "action_min": float(tensors["actions"].min()),
        "action_max": float(tensors["actions"].max()),
        "action_abs_gt_0p95_fraction": float((tensors["actions"].abs() > 0.95).float().mean()),
        "simulator_snapshots": {
            "present": True,
            "steps": trajectory_length,
            "rows_per_step": num_trajectories,
            "required_keys": list(SNAPSHOT_REQUIRED_KEYS),
            "dataset_keys": dict(SNAPSHOT_DATASET_KEYS),
            "snapshot_version": (data.get("metadata") or {}).get(
                "simulator_snapshot_version"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
