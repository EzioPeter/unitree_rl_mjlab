"""Select 25 complete environment trajectories from a rectangular Go2 dataset.

The input is expected to contain T x E transitions with T=1000.  Selection is
performed at environment-column granularity: a selected column is never cut or
spliced.  A MILP chooses 25 columns whose aggregate command-mode and signed
magnitude-bin counts best match the requested distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


MODE_NAMES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
MODE_BY_MASK = {
    (False, False, False): 0,
    (True, False, False): 1,
    (False, True, False): 2,
    (False, False, True): 3,
    (True, True, False): 4,
    (True, False, True): 5,
    (False, True, True): 6,
    (True, True, True): 7,
}
MODE_TARGETS = np.asarray((2000, 6250, 2500, 2000, 3500, 4250, 1250, 3250), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--num-trajectories", type=int, default=25)
    parser.add_argument("--trajectory-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.08, 0.32, 0.56, 0.80))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.05, 0.133333, 0.216667, 0.30))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.08, 0.253333, 0.426667, 0.60))
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument(
        "--required-selection-report",
        default=None,
        help=(
            "Optional earlier selection report. Every selected_env_id in that "
            "report is forced into this selection, enabling nested 25K->50K "
            "data-size ablations."
        ),
    )
    parser.add_argument(
        "--allow-mid-trajectory-termination",
        action="store_true",
        help="Allow environment columns containing a termination before the final timestep.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stack_time(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.stack(value)


def select_env_rows(value: Any, index: torch.Tensor, num_source_envs: int) -> Any:
    """Recursively select environment rows, including nested TRACE snapshots."""

    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == num_source_envs:
            return value.index_select(0, index).clone()
        return value.clone()
    if isinstance(value, dict):
        return {
            key: select_env_rows(item, index, num_source_envs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [select_env_rows(item, index, num_source_envs) for item in value]
    if isinstance(value, tuple):
        return tuple(select_env_rows(item, index, num_source_envs) for item in value)
    return copy.deepcopy(value)


def command_features(
    commands: torch.Tensor,
    *,
    epsilon: float,
    edges: tuple[tuple[float, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-environment mode counts [E,8] and marginal counts [E,18]."""
    cmd = commands.detach().cpu().numpy()
    active = np.abs(cmd) > epsilon
    time_steps, num_envs, _ = cmd.shape
    modes = np.empty((time_steps, num_envs), dtype=np.int8)
    for mask, mode_id in MODE_BY_MASK.items():
        matches = np.ones((time_steps, num_envs), dtype=bool)
        for axis in range(3):
            matches &= active[..., axis] == mask[axis]
        modes[matches] = mode_id
    mode_counts = np.stack(
        [(modes == mode_id).sum(axis=0) for mode_id in range(len(MODE_NAMES))],
        axis=1,
    ).astype(np.float64)

    marginal_counts = np.zeros((num_envs, 18), dtype=np.float64)
    for axis in range(3):
        magnitude = np.abs(cmd[..., axis])
        axis_edges = np.asarray(edges[axis], dtype=np.float64)
        # Three equal-width magnitude bins within the configured non-zero range.
        bins = np.digitize(magnitude, axis_edges[1:-1], right=False)
        valid = active[..., axis] & (magnitude >= axis_edges[0] - 1.0e-6)
        valid &= magnitude <= axis_edges[-1] + 1.0e-6
        for sign_id, sign_mask in enumerate((cmd[..., axis] < 0.0, cmd[..., axis] > 0.0)):
            for bin_id in range(3):
                feature_id = axis * 6 + sign_id * 3 + bin_id
                marginal_counts[:, feature_id] = (
                    valid & sign_mask & (bins == bin_id)
                ).sum(axis=0)
    return mode_counts, marginal_counts


def marginal_targets(mode_targets: np.ndarray = MODE_TARGETS) -> np.ndarray:
    # Expected active-axis counts implied by the eight command-mode targets.
    x_active = mode_targets[[1, 4, 5, 7]].sum()
    y_active = mode_targets[[2, 4, 6, 7]].sum()
    yaw_active = mode_targets[[3, 5, 6, 7]].sum()
    return np.concatenate(
        (
            np.full(6, x_active / 6.0),
            np.full(6, y_active / 6.0),
            np.full(6, yaw_active / 6.0),
        )
    )


def select_envs(
    mode_counts: np.ndarray,
    marginal_counts: np.ndarray,
    *,
    count: int,
    seed: int,
    time_limit: float,
    target_transitions: int,
    required_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_scale = float(target_transitions) / float(MODE_TARGETS.sum())
    mode_targets = MODE_TARGETS * target_scale
    features = np.concatenate((mode_counts, marginal_counts), axis=1)
    targets = np.concatenate((mode_targets, marginal_targets(mode_targets)))
    num_envs, num_features = features.shape
    variables = num_envs + 2 * num_features

    # features.T @ selected - positive_deviation + negative_deviation = target
    equality = np.zeros((num_features + 1, variables), dtype=np.float64)
    equality[:num_features, :num_envs] = features.T
    equality[:num_features, num_envs : num_envs + num_features] = -np.eye(num_features)
    equality[:num_features, num_envs + num_features :] = np.eye(num_features)
    equality[-1, :num_envs] = 1.0
    lower = np.concatenate((targets, [float(count)]))
    upper = lower.copy()

    objective = np.zeros(variables, dtype=np.float64)
    rng = np.random.default_rng(seed)
    objective[:num_envs] = rng.uniform(0.0, 1.0e-10, size=num_envs)
    # Command-mode agreement dominates; signed magnitude balance is secondary.
    weights = np.concatenate((10.0 / mode_targets, 1.0 / marginal_targets(mode_targets)))
    objective[num_envs : num_envs + num_features] = weights
    objective[num_envs + num_features :] = weights

    result = milp(
        c=objective,
        integrality=np.concatenate((np.ones(num_envs), np.zeros(2 * num_features))),
        bounds=Bounds(
            np.concatenate(
                (
                    np.where(
                        np.isin(
                            np.arange(num_envs),
                            np.asarray(
                                [] if required_indices is None else required_indices,
                                dtype=np.int64,
                            ),
                        ),
                        1.0,
                        0.0,
                    ),
                    np.zeros(2 * num_features),
                )
            ),
            np.concatenate((np.ones(num_envs), np.full(2 * num_features, np.inf))),
        ),
        constraints=LinearConstraint(csr_matrix(equality), lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"Full-trajectory MILP failed: status={result.status}, {result.message}")
    selected = np.flatnonzero(result.x[:num_envs] > 0.5)
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} selected trajectories, got {len(selected)}")
    selected_modes = mode_counts[selected].sum(axis=0)
    selected_marginals = marginal_counts[selected].sum(axis=0)
    diagnostics = {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun),
        "selected_env_ids": selected.astype(int).tolist(),
        "actual_mode_counts": {
            name: int(value) for name, value in zip(MODE_NAMES, selected_modes, strict=True)
        },
        "desired_mode_counts": {
            name: int(round(value))
            for name, value in zip(MODE_NAMES, mode_targets, strict=True)
        },
        "mode_relative_errors": {
            name: float((actual - target) / target)
            for name, actual, target in zip(MODE_NAMES, selected_modes, mode_targets, strict=True)
        },
        "actual_marginal_counts": selected_marginals.astype(int).tolist(),
        "desired_marginal_counts": marginal_targets(mode_targets).tolist(),
        "target_scale_from_25k": target_scale,
        "required_indices": (
            []
            if required_indices is None
            else np.asarray(required_indices, dtype=np.int64).astype(int).tolist()
        ),
    }
    return selected, diagnostics


def select_columns(data: dict[str, Any], env_ids: np.ndarray, time_steps: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    index = torch.as_tensor(env_ids, dtype=torch.long)
    num_source_envs = int(data["num_envs"])
    for key, value in data.items():
        if isinstance(value, list) and len(value) == time_steps:
            output[key] = [
                select_env_rows(row, index, num_source_envs) for row in value
            ]
        elif isinstance(value, torch.Tensor) and value.ndim >= 2:
            if value.shape[0] == time_steps and value.shape[1] == num_source_envs:
                output[key] = value.index_select(1, index).clone()
            else:
                output[key] = value.clone()
        elif key == "metadata":
            output[key] = copy.deepcopy(value or {})
        elif key not in ("startup_domain_table", "episode_domain_table"):
            output[key] = copy.deepcopy(value)

    startup = data.get("startup_domain_table")
    if isinstance(startup, dict):
        filtered = {}
        for key, value in startup.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == num_source_envs:
                filtered[key] = value.index_select(0, index).clone()
            else:
                filtered[key] = copy.deepcopy(value)
        if "startup_domain_id" in filtered:
            filtered["startup_domain_id"] = torch.arange(len(env_ids), dtype=torch.long)
        output["startup_domain_table"] = filtered

    episode_table = data.get("episode_domain_table")
    if isinstance(episode_table, dict) and "startup_domain_id" in episode_table:
        source_ids = episode_table["startup_domain_id"].long()
        keep = torch.zeros_like(source_ids, dtype=torch.bool)
        remapped = torch.full_like(source_ids, -1)
        for new_id, old_id in enumerate(env_ids.tolist()):
            matches = source_ids == int(old_id)
            keep |= matches
            remapped[matches] = int(new_id)
        filtered = {}
        for key, value in episode_table.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == len(source_ids):
                filtered[key] = value[keep].clone()
            else:
                filtered[key] = copy.deepcopy(value)
        filtered["startup_domain_id"] = remapped[keep]
        output["episode_domain_table"] = filtered

    output["num_envs"] = int(len(env_ids))
    output["capacity"] = int(time_steps * len(env_ids))
    output["source_env_ids"] = [
        torch.as_tensor(env_ids, dtype=torch.long).clone() for _ in range(time_steps)
    ]
    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    commands = stack_time(data["commands"])
    if commands.ndim != 3 or commands.shape[-1] != 3:
        raise ValueError(f"Expected commands [T,E,3], got {tuple(commands.shape)}")
    time_steps, num_envs = commands.shape[:2]
    if time_steps != args.trajectory_length:
        raise ValueError(
            f"Expected exactly {args.trajectory_length} steps per environment, got {time_steps}"
        )
    if num_envs < args.num_trajectories:
        raise ValueError(f"Only {num_envs} environment trajectories are available")

    edges = (tuple(args.x_edges), tuple(args.y_edges), tuple(args.yaw_edges))
    mode_counts, marginal_counts = command_features(
        commands,
        epsilon=float(args.zero_epsilon),
        edges=edges,
    )
    terminations = stack_time(data["terminations"])
    if terminations.ndim == 3 and terminations.shape[-1] == 1:
        terminations = terminations[..., 0]
    if terminations.shape != (time_steps, num_envs):
        raise ValueError(
            f"Expected terminations [T,E] or [T,E,1], got {tuple(terminations.shape)}"
        )
    if args.allow_mid_trajectory_termination:
        eligible = np.arange(num_envs, dtype=np.int64)
    else:
        mid_termination = (terminations[:-1] > 0.5).any(dim=0)
        eligible = torch.nonzero(~mid_termination, as_tuple=False).flatten().cpu().numpy()
    if len(eligible) < args.num_trajectories:
        raise RuntimeError(
            "Not enough uninterrupted environment trajectories: "
            f"need {args.num_trajectories}, found {len(eligible)}"
        )
    required_source_ids: list[int] = []
    required_local = np.empty(0, dtype=np.int64)
    if args.required_selection_report:
        required_report_path = Path(args.required_selection_report).expanduser().resolve()
        required_report = json.loads(required_report_path.read_text())
        required_source_ids = [int(value) for value in required_report["selected_env_ids"]]
        source_to_local = {int(source_id): local for local, source_id in enumerate(eligible.tolist())}
        missing_required = [
            source_id for source_id in required_source_ids if source_id not in source_to_local
        ]
        if missing_required:
            raise RuntimeError(
                f"Required source trajectories are not eligible: {missing_required}"
            )
        required_local = np.asarray(
            [source_to_local[source_id] for source_id in required_source_ids],
            dtype=np.int64,
        )
        if len(required_local) > int(args.num_trajectories):
            raise ValueError(
                "Required selection is larger than the requested trajectory count."
            )

    selected_local, diagnostics = select_envs(
        mode_counts[eligible],
        marginal_counts[eligible],
        count=int(args.num_trajectories),
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        target_transitions=int(args.num_trajectories) * int(time_steps),
        required_indices=required_local,
    )
    selected = eligible[selected_local]
    diagnostics["eligible_uninterrupted_trajectories"] = int(len(eligible))
    diagnostics["excluded_mid_termination_trajectories"] = int(num_envs - len(eligible))
    diagnostics["selected_env_ids"] = selected.astype(int).tolist()
    diagnostics["required_source_env_ids"] = required_source_ids
    diagnostics["nested_selection"] = bool(args.required_selection_report)
    diagnostics["required_selection_report"] = (
        str(Path(args.required_selection_report).expanduser().resolve())
        if args.required_selection_report
        else None
    )
    output = select_columns(data, selected, time_steps)
    output["metadata"].update(
        {
            "condition_id": str(args.condition_id),
            "selection_kind": (
                "failure_aware_environment_columns_v1"
                if args.allow_mid_trajectory_termination
                else "complete_environment_trajectories_v1"
            ),
            "selection_source": str(input_path),
            "selection_source_sha256": sha256(input_path),
            "selection_seed": int(args.seed),
            "selected_source_env_ids": selected.astype(int).tolist(),
            "num_time_steps": int(time_steps),
            "num_envs": int(len(selected)),
            "num_transitions": int(time_steps * len(selected)),
            "actual_num_transitions": int(time_steps * len(selected)),
            "trajectory_length": int(time_steps),
            "whole_trajectory_selection": not bool(
                args.allow_mid_trajectory_termination
            ),
            "failure_aware_selection": bool(
                args.allow_mid_trajectory_termination
            ),
            "mid_trajectory_terminations_allowed": bool(
                args.allow_mid_trajectory_termination
            ),
            "selection_mode_counts": diagnostics["actual_mode_counts"],
            "selection_desired_mode_counts": diagnostics["desired_mode_counts"],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    terminations = stack_time(output["terminations"])
    episode_ids = stack_time(output["episode_ids"])
    report = {
        "schema": (
            "go2_failure_aware_environment_column_selection_v1"
            if args.allow_mid_trajectory_termination
            else "go2_complete_environment_trajectory_selection_v1"
        ),
        "condition_id": str(args.condition_id),
        "input": str(input_path),
        "input_sha256": output["metadata"]["selection_source_sha256"],
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "source_shape": [int(time_steps), int(num_envs)],
        "selected_shape": [int(time_steps), int(len(selected))],
        "selected_transitions": int(time_steps * len(selected)),
        "whole_trajectory_selection": not bool(
            args.allow_mid_trajectory_termination
        ),
        "failure_aware_selection": bool(
            args.allow_mid_trajectory_termination
        ),
        "termination_count": int((terminations > 0.5).sum()),
        "unique_episode_ids": int(torch.unique(episode_ids).numel()),
        "magnitude_bin_edges": {
            "x": list(args.x_edges),
            "y": list(args.y_edges),
            "yaw": list(args.yaw_edges),
        },
        **diagnostics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
