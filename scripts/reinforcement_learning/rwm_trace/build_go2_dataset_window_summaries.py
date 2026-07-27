#!/usr/bin/env python3
"""Build task-specific offline windows for the initial Go2 TRACE scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.trajectory import summarize_go2_trajectory


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--condition_id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_length", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    return parser.parse_args()


def _stack(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        result = value
    elif isinstance(value, list) and value and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        result = torch.stack(value)
    else:
        raise ValueError(f"Dataset field {key!r} must be a tensor or tensor list.")
    if result.ndim < 2:
        raise ValueError(f"Dataset field {key!r} must begin [time,env].")
    return result.detach().cpu()


def main() -> None:
    args = _args()
    if args.window_length < 2 or args.stride < 1:
        raise ValueError("window_length must be >=2 and stride positive.")
    dataset_path = Path(args.dataset).expanduser().resolve()
    config_path = Path(args.training_config).expanduser().resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    required = (
        "states",
        "actions",
        "next_states",
        "contacts",
        "terminations",
        "commands",
        "rewards",
        "prev_actions",
        "episode_ids",
        "timesteps",
    )
    tensors = {key: _stack(dataset, key) for key in required}
    prefix = tensors["states"].shape[:2]
    if any(value.shape[:2] != prefix for value in tensors.values()):
        raise ValueError("Offline dataset fields have inconsistent [time,env] dimensions.")
    time_steps, num_envs = map(int, prefix)

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    world = cfg.world_model
    thresholds = (
        float(world.reward_command_active_threshold_x),
        float(world.reward_command_active_threshold_y),
        float(world.reward_command_active_threshold_yaw),
    )
    floor = float(world.reward_response_command_scale_floor)
    namespace = sha256_path(dataset_path)[:16]
    root = (
        _stack(dataset, "sim_root_states_local")
        if dataset.get("sim_root_states_local") is not None
        else None
    )
    feet = (
        _stack(dataset, "trace_foot_site_positions_w")
        if dataset.get("trace_foot_site_positions_w") is not None
        else None
    )
    foot_vel = (
        _stack(dataset, "trace_foot_site_linear_velocities_w")
        if dataset.get("trace_foot_site_linear_velocities_w") is not None
        else None
    )

    summaries: list[dict[str, Any]] = []
    for env_index in range(num_envs):
        segment_start = 0
        while segment_start < time_steps:
            segment_stop = segment_start + 1
            while segment_stop < time_steps:
                same_episode = (
                    tensors["episode_ids"][segment_stop, env_index]
                    == tensors["episode_ids"][segment_stop - 1, env_index]
                )
                contiguous = (
                    tensors["timesteps"][segment_stop, env_index]
                    == tensors["timesteps"][segment_stop - 1, env_index] + 1
                )
                if not bool(same_episode and contiguous):
                    break
                segment_stop += 1
            for start in range(
                segment_start,
                segment_stop - args.window_length + 1,
                args.stride,
            ):
                stop = start + args.window_length
                episode = int(tensors["episode_ids"][start, env_index])
                timestep = int(tensors["timesteps"][start, env_index])
                trajectory: dict[str, Any] = {
                    key: value[start:stop, env_index]
                    for key, value in tensors.items()
                    if key not in {"episode_ids", "timesteps"}
                }
                trajectory.update(
                    {
                        "trajectory_id": (
                            f"{namespace}_env{env_index:05d}_ep{episode:08d}"
                            f"_t{timestep:06d}"
                        ),
                        "start_state_id": f"{env_index}:{start}",
                        "start_state_key": f"{namespace}:{env_index}:{start}",
                        "comparison_group_key": f"{namespace}:episode:{episode}",
                        "source_kind": "offline_window",
                        "task_id": args.task_id,
                        "dataset_id": args.dataset_id,
                        "condition_id": args.condition_id,
                        "step_dt": float(world.step_dt),
                        "expected_trajectory_length": args.window_length,
                        "command_active_thresholds": thresholds,
                        "command_normalization_floors": (floor, floor, floor),
                        "action_saturation_threshold": float(
                            world.reward_action_saturation_threshold
                        ),
                    }
                )
                if root is not None:
                    trajectory["sim_root_states_local"] = root[start:stop, env_index]
                if feet is not None and foot_vel is not None:
                    trajectory["trace_foot_site_positions_w"] = feet[start:stop, env_index]
                    trajectory["trace_foot_site_linear_velocities_w"] = foot_vel[
                        start:stop, env_index
                    ]
                summaries.append(summarize_go2_trajectory(trajectory))
            segment_start = segment_stop

    if len(summaries) < 2:
        raise RuntimeError("Dataset produced fewer than two valid offline windows.")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(
                json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "dataset_sha256": sha256_path(dataset_path),
                "training_config": str(config_path),
                "training_config_sha256": sha256_path(config_path),
                "task_id": args.task_id,
                "condition_id": args.condition_id,
                "summary_count": len(summaries),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
