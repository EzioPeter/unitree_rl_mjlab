"""Slice a vectorized Go2 RWM dataset by complete env trajectories.

The collected RWM dataset is stored as a time-major list of tensors with shape
``[num_envs, ...]``. This utility keeps a subset of env columns intact so RNN
world-model training still sees contiguous trajectories rather than shuffled
transitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import (  # noqa: E402
    COLLECTOR_ID_TO_NAME,
    load_mixed_dataset,
    save_dataset_dict,
)


TIME_MAJOR_KEYS = {
    "states",
    "actions",
    "next_states",
    "contacts",
    "terminations",
    "observations",
    "next_observations",
    "commands",
    "rewards",
    "dones",
    "timeouts",
    "prev_actions",
    "episode_ids",
    "timesteps",
    "collector_types",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_trajectories", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=("stratified", "random", "first"), default="stratified")
    return parser.parse_args()


def _stack_key(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset[key]
    if isinstance(value, torch.Tensor):
        return value
    return torch.stack(value, dim=0)


def _env_collector_modes(dataset: dict[str, Any]) -> torch.Tensor:
    collector = _stack_key(dataset, "collector_types").long()
    num_envs = int(collector.shape[1])
    modes = torch.zeros(num_envs, dtype=torch.long)
    for env_id in range(num_envs):
        values, counts = collector[:, env_id].unique(return_counts=True)
        modes[env_id] = values[counts.argmax()]
    return modes


def _allocate_counts(global_counts: Counter[int], total: int) -> dict[int, int]:
    count_sum = sum(global_counts.values())
    raw = {key: total * value / count_sum for key, value in global_counts.items()}
    alloc = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(alloc.values())
    order = sorted(raw, key=lambda key: raw[key] - alloc[key], reverse=True)
    for key in order[:remaining]:
        alloc[key] += 1
    return alloc


def _select_env_ids(dataset: dict[str, Any], num_trajectories: int, seed: int, selection: str) -> torch.Tensor:
    num_envs = int(dataset["num_envs"])
    if num_trajectories <= 0 or num_trajectories > num_envs:
        raise ValueError(f"num_trajectories must be in [1, {num_envs}], got {num_trajectories}.")
    generator = torch.Generator().manual_seed(int(seed))
    if selection == "first":
        return torch.arange(num_trajectories, dtype=torch.long)
    if selection == "random":
        return torch.randperm(num_envs, generator=generator)[:num_trajectories].sort().values

    modes = _env_collector_modes(dataset)
    global_counts = Counter(int(v) for v in modes.tolist())
    alloc = _allocate_counts(global_counts, num_trajectories)
    selected: list[int] = []
    for collector_id in sorted(alloc):
        candidates = (modes == collector_id).nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0 or alloc[collector_id] <= 0:
            continue
        perm = torch.randperm(candidates.numel(), generator=generator)
        selected.extend(candidates[perm[: alloc[collector_id]]].tolist())
    if len(selected) < num_trajectories:
        already = torch.tensor(selected, dtype=torch.long)
        mask = torch.ones(num_envs, dtype=torch.bool)
        if already.numel() > 0:
            mask[already] = False
        remaining = mask.nonzero(as_tuple=False).flatten()
        perm = torch.randperm(remaining.numel(), generator=generator)
        selected.extend(remaining[perm[: num_trajectories - len(selected)]].tolist())
    return torch.tensor(selected[:num_trajectories], dtype=torch.long).sort().values


def _slice_time_major(value: Any, env_ids: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        return value[:, env_ids].clone()
    if isinstance(value, list):
        return [item.index_select(0, env_ids).clone() for item in value]
    return value


def main() -> None:
    args = _parse_args()
    source_path = Path(args.source_path).expanduser()
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    save_path = Path(args.save_path).expanduser()
    if not save_path.is_absolute():
        save_path = REPO_ROOT / save_path

    dataset = load_mixed_dataset(source_path)
    env_ids = _select_env_ids(dataset, int(args.num_trajectories), int(args.seed), str(args.selection))
    sliced: dict[str, Any] = {}
    for key, value in dataset.items():
        if key in TIME_MAJOR_KEYS:
            sliced[key] = _slice_time_major(value, env_ids)
        else:
            sliced[key] = value

    num_time_steps = len(sliced["states"]) if isinstance(sliced["states"], list) else int(sliced["states"].shape[0])
    num_envs = int(env_ids.numel())
    metadata = dict(dataset.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": str(source_path),
            "subset_kind": "env_trajectories",
            "selection": str(args.selection),
            "seed": int(args.seed),
            "selected_env_ids": env_ids.tolist(),
            "num_trajectories": num_envs,
            "num_time_steps": num_time_steps,
            "num_transitions": num_time_steps * num_envs,
            "actual_num_transitions": num_time_steps * num_envs,
        }
    )
    sliced["metadata"] = metadata
    sliced["num_envs"] = num_envs
    sliced["capacity"] = num_time_steps * num_envs

    save_dataset_dict(sliced, save_path)
    modes = _env_collector_modes(sliced)
    mode_counts = Counter(int(v) for v in modes.tolist())
    named_counts = {COLLECTOR_ID_TO_NAME.get(key, str(key)): value for key, value in sorted(mode_counts.items())}
    print(f"[Go2-DatasetSlice] source={source_path}")
    print(f"[Go2-DatasetSlice] save={save_path}")
    print(f"[Go2-DatasetSlice] selected_env_ids={env_ids.tolist()}")
    print(f"[Go2-DatasetSlice] trajectories={num_envs}, time_steps={num_time_steps}, transitions={num_time_steps * num_envs}")
    print(f"[Go2-DatasetSlice] collector_mode_counts={json.dumps(named_counts, sort_keys=True)}")


if __name__ == "__main__":
    main()
