#!/usr/bin/env python3
"""Relabel an existing canonical real replay with aligned task shaping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.reinforcement_learning.rwm_flashsac.build_go2_real_replay import (
    _stack_time_key,
)
from scripts.reinforcement_learning.rwm_trace.multi_axis_reward_shaping import (
    aligned_multi_axis_quality_delta_torch,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_replay", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("soft_gaussian_advantage", "corrected_half_tanh"),
    )
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--tanh_gain", type=float, default=2.0)
    parser.add_argument("--overspeed_weight", type=float, default=4.0)
    parser.add_argument("--lateral_axis_weight", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_replay).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    replay = torch.load(input_path, map_location="cpu", weights_only=False)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    commands = _stack_time_key(dataset, "commands").float()
    next_states = _stack_time_key(dataset, "next_states").float()
    velocity = next_states[..., [0, 1, 5]]
    shaping = {
        "mode": str(args.mode),
        "weight": float(args.weight),
        "active_thresholds": [0.03, 0.02, 0.03],
        "axis_weights": [1.0, float(args.lateral_axis_weight), 1.0],
        "command_scale_floor": 0.05,
        "tracking_stds": [0.25, 0.10, 0.20],
        "tanh_gain": float(args.tanh_gain),
        "overspeed_weight": float(args.overspeed_weight),
        "step_dt": 0.02,
    }
    one_step_delta = aligned_multi_axis_quality_delta_torch(
        velocity=velocity,
        command=commands,
        active_thresholds=tuple(shaping["active_thresholds"]),
        axis_weights=tuple(shaping["axis_weights"]),
        command_scale_floor=float(shaping["command_scale_floor"]),
        tracking_stds=tuple(shaping["tracking_stds"]),
        mode=str(shaping["mode"]),
        weight=float(shaping["weight"]),
        tanh_gain=float(shaping["tanh_gain"]),
        overspeed_weight=float(shaping["overspeed_weight"]),
    )

    time_index = replay["source_time_index"].long()
    env_index = replay["source_env_index"].long()
    lengths = replay["n_step_length"].long()
    gamma = float(replay["metadata"]["gamma"])
    n_step = int(replay["metadata"]["n_step"])
    n_step_delta = torch.zeros_like(replay["reward"])
    for offset in range(n_step):
        included = lengths > offset
        n_step_delta[included] += (
            gamma**offset
            * one_step_delta[
                time_index[included] + offset,
                env_index[included],
            ]
        )
    step_dt = float(shaping["step_dt"])
    replay["reward"] = (
        replay["reward"].float() + step_dt * n_step_delta
    ).contiguous()
    replay["one_step_reward"] = (
        replay["one_step_reward"].float()
        + step_dt * one_step_delta[time_index, env_index]
    ).contiguous()

    shaping_json = json.dumps(
        shaping, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    replay["metadata"] = {
        **replay["metadata"],
        "base_real_replay_path": str(input_path),
        "base_real_replay_sha256": _sha256_file(input_path),
        "aligned_task_reward": shaping,
        "aligned_task_reward_sha256": hashlib.sha256(
            shaping_json
        ).hexdigest(),
        "n_step_reward_mean": float(replay["reward"].mean()),
        "n_step_reward_std": float(
            replay["reward"].std(unbiased=False)
        ),
        "one_step_reward_mean": float(replay["one_step_reward"].mean()),
        "one_step_reward_std": float(
            replay["one_step_reward"].std(unbiased=False)
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(replay, temporary_path)
    temporary_path.replace(output_path)
    print(json.dumps(replay["metadata"], indent=2, sort_keys=True))
    print(f"[aligned real replay] wrote {output_path}")
    print(f"[aligned real replay] sha256={_sha256_file(output_path)}")


if __name__ == "__main__":
    main()
