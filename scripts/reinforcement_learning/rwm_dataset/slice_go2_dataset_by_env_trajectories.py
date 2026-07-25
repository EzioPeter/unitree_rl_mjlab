"""Select intact environment columns from a mixed Go2 RWM dataset.

V13 extends the upstream stratified selector so every time-major field is
selected together, including simulator snapshots and action histories needed
for TRACE exact reset.  Unknown time-major fields are selected by shape rather
than silently copied with the source environment count.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    COLLECTOR_NAME_TO_ID,
    load_mixed_dataset,
    save_dataset_dict,
)


V13_COLLECTOR_MIX = {
    "expert": 0.45,
    "noisy_expert": 0.25,
    "medium": 0.10,
    "failure_border": 0.15,
    "random": 0.05,
}
V13_SNAPSHOT_WIDTHS = {
    "sim_root_states_local": 13,
    "sim_joint_positions": 12,
    "sim_joint_velocities": 12,
    "sim_action_histories": 12,
    "sim_prev_action_histories": 12,
    "sim_prev_prev_action_histories": 12,
    "sim_snapshot_commands": 3,
}
V13_CONDITION_TARGETS = {
    "g0": {"rr_calf_strength_range": [1.0, 1.0], "payload_mass_range_kg": [0.0, 0.0]},
    "rr05": {"rr_calf_strength_range": [0.5, 0.5], "payload_mass_range_kg": [0.0, 0.0]},
    "rr03": {"rr_calf_strength_range": [0.3, 0.3], "payload_mass_range_kg": [0.0, 0.0]},
    "p5": {"rr_calf_strength_range": [1.0, 1.0], "payload_mass_range_kg": [5.0, 5.0]},
    "p75": {"rr_calf_strength_range": [1.0, 1.0], "payload_mass_range_kg": [7.5, 7.5]},
}
V13_COMMAND_MODES = [
    "stand",
    "pure_x",
    "pure_y",
    "pure_yaw",
    "xy",
    "x_yaw",
    "y_yaw",
    "xy_yaw",
]
V13_COMMAND_WEIGHTS = {
    "stand": 0.08,
    "pure_x": 0.25,
    "pure_y": 0.10,
    "pure_yaw": 0.08,
    "xy": 0.14,
    "x_yaw": 0.17,
    "y_yaw": 0.05,
    "xy_yaw": 0.13,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--condition_id", required=True)
    parser.add_argument("--expected_expert_policy_path", required=True)
    parser.add_argument("--launch_manifest_path", required=True)
    parser.add_argument("--num_trajectories", type=int, default=25)
    parser.add_argument("--expected_time_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("stratified", "random", "first"),
        default="stratified",
    )
    parser.add_argument(
        "--require_trace_snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack_key(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(f"Dataset key {key!r} must be a non-empty tensor/list.")
    return torch.stack(value, dim=0)


def _normalized_mix(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("Dataset metadata is missing collector_mix.")
    unknown = sorted(set(value) - set(V13_COLLECTOR_MIX))
    if unknown:
        raise ValueError(f"Unknown collectors in metadata: {unknown}")
    result = {name: float(value.get(name, 0.0)) for name in V13_COLLECTOR_MIX}
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("collector_mix must sum to a positive value.")
    return {name: weight / total for name, weight in result.items()}


def _validate_requested_mix(dataset: dict[str, Any]) -> dict[str, float]:
    metadata = dict(dataset.get("metadata") or {})
    actual = _normalized_mix(metadata.get("collector_mix"))
    for name, expected in V13_COLLECTOR_MIX.items():
        if abs(actual[name] - expected) > 1.0e-9:
            raise ValueError(
                f"V13 collector mix mismatch for {name}: "
                f"metadata={actual[name]}, expected={expected}"
            )
    return actual


def _require_close_sequence(
    actual: Any,
    expected: list[float],
    label: str,
    *,
    tolerance: float = 1.0e-6,
) -> None:
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise ValueError(f"{label}={actual!r}, expected {expected!r}.")
    if any(abs(float(left) - float(right)) > tolerance for left, right in zip(actual, expected)):
        raise ValueError(f"{label}={actual!r}, expected {expected!r}.")


def _validate_source_protocol(
    dataset: dict[str, Any],
    *,
    condition_id: str,
    expected_expert_policy_path: str,
) -> dict[str, Any]:
    if condition_id not in V13_CONDITION_TARGETS:
        raise ValueError(f"Unknown V13 condition {condition_id!r}.")
    metadata = dict(dataset.get("metadata") or {})
    checks = {
        "task": "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
        "seed": 0,
        "obs_dim": 45,
        "dataset_obs_kind": "proprioceptive",
        "state_dim": 45,
        "action_dim": 12,
        "live_actor_obs_dim": 45,
        "live_critic_obs_dim": 48,
        "num_envs": 1024,
        "requested_num_transitions": 1_024_000,
        "fixed_collector_assignment": False,
        "medium_policy_path": None,
        "save_trace_snapshots": True,
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"metadata.{key}={actual!r}, expected {expected!r}.")
    for key, expected in {
        "action_noise_std": 0.12,
        "medium_action_noise_std": 0.25,
        "failure_action_noise_std": 0.45,
    }.items():
        if abs(float(metadata.get(key, float("nan"))) - expected) > 1.0e-9:
            raise ValueError(f"metadata.{key}={metadata.get(key)!r}, expected {expected}.")
    if int(dataset.get("num_envs", -1)) != 1024:
        raise ValueError(f"dataset.num_envs={dataset.get('num_envs')!r}, expected 1024.")
    if int(metadata.get("actual_num_transitions", -1)) != 1_024_000:
        raise ValueError(
            f"metadata.actual_num_transitions={metadata.get('actual_num_transitions')!r}, "
            "expected 1024000."
        )
    actual_expert = str(Path(str(metadata.get("expert_policy_path"))).expanduser().resolve())
    expected_expert = str(Path(expected_expert_policy_path).expanduser().resolve())
    if actual_expert != expected_expert:
        raise ValueError(
            f"metadata.expert_policy_path={actual_expert!r}, expected {expected_expert!r}."
        )

    dagger = dict(metadata.get("dagger_relabel") or {})
    if bool(dagger.get("enabled")):
        raise ValueError("V13 mixed collection forbids expert action relabeling.")
    if metadata.get("command_modes") != V13_COMMAND_MODES:
        raise ValueError(
            f"metadata.command_modes={metadata.get('command_modes')!r}, "
            f"expected {V13_COMMAND_MODES!r}."
        )
    weights = dict(metadata.get("command_mode_weights") or {})
    for name, expected in V13_COMMAND_WEIGHTS.items():
        if abs(float(weights.get(name, float("nan"))) - expected) > 1.0e-6:
            raise ValueError(
                f"metadata.command_mode_weights[{name!r}]={weights.get(name)!r}, "
                f"expected {expected}."
            )
    command_ranges = dict(metadata.get("command_ranges") or {})
    _require_close_sequence(command_ranges.get("x_range"), [-0.5, 0.5], "command_ranges.x_range")
    if command_ranges.get("signed_x") is not True:
        raise ValueError("command_ranges.signed_x must be true.")
    _require_close_sequence(
        command_ranges.get("x_abs_range"), [0.05, 0.5], "command_ranges.x_abs_range"
    )
    _require_close_sequence(
        command_ranges.get("y_abs_range"), [0.03, 0.2], "command_ranges.y_abs_range"
    )
    _require_close_sequence(
        command_ranges.get("yaw_abs_range"), [0.05, 0.4], "command_ranges.yaw_abs_range"
    )
    _require_close_sequence(
        metadata.get("command_resample_interval"),
        [120.0, 300.0],
        "metadata.command_resample_interval",
    )

    env_randomization = dict(metadata.get("env_randomization") or {})
    for key in ("use_domain_randomization", "use_push_randomization", "use_observation_noise"):
        if bool(env_randomization.get(key)):
            raise ValueError(f"metadata.env_randomization.{key} must be false.")
    action_interface = dict(metadata.get("env_step_action_interface") or {})
    if float(action_interface.get("env_action_noise_std", float("nan"))) != 0.0:
        raise ValueError("env_action_noise_std must be 0.")
    if float(action_interface.get("env_action_bias_std", float("nan"))) != 0.0:
        raise ValueError("env_action_bias_std must be 0.")
    _require_close_sequence(
        action_interface.get("env_action_scale_range"),
        [1.0, 1.0],
        "env_action_scale_range",
    )
    _require_close_sequence(
        action_interface.get("env_action_delay_steps"),
        [0.0, 0.0],
        "env_action_delay_steps",
    )

    target_gap = dict(metadata.get("target_gap") or {})
    for key, expected in V13_CONDITION_TARGETS[condition_id].items():
        _require_close_sequence(target_gap.get(key), expected, f"target_gap.{key}")
    return metadata


def _env_collector_modes(dataset: dict[str, Any]) -> torch.Tensor:
    collector = _stack_key(dataset, "collector_types").long()
    if collector.ndim != 2:
        raise ValueError(f"collector_types must have shape [T,E], got {tuple(collector.shape)}")
    num_envs = int(collector.shape[1])
    modes = torch.zeros(num_envs, dtype=torch.long)
    allowed_ids = set(COLLECTOR_ID_TO_NAME)
    present_ids = set(int(value) for value in torch.unique(collector).tolist())
    if not present_ids.issubset(allowed_ids):
        raise ValueError(f"collector_types contains unknown IDs: {sorted(present_ids - allowed_ids)}")
    for env_id in range(num_envs):
        values, counts = collector[:, env_id].unique(return_counts=True)
        modes[env_id] = values[counts.argmax()]
    return modes


def _allocate_counts(global_counts: Counter[int], total: int) -> dict[int, int]:
    count_sum = sum(global_counts.values())
    raw = {key: total * value / count_sum for key, value in global_counts.items()}
    allocated = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        raw,
        key=lambda key: (raw[key] - allocated[key], -key),
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _select_env_ids(
    dataset: dict[str, Any],
    num_trajectories: int,
    seed: int,
    selection: str,
) -> torch.Tensor:
    num_envs = int(dataset["num_envs"])
    if num_trajectories <= 0 or num_trajectories > num_envs:
        raise ValueError(
            f"num_trajectories must be in [1, {num_envs}], got {num_trajectories}."
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if selection == "first":
        return torch.arange(num_trajectories, dtype=torch.long)
    if selection == "random":
        return torch.randperm(num_envs, generator=generator)[:num_trajectories].sort().values

    modes = _env_collector_modes(dataset)
    global_counts = Counter(int(value) for value in modes.tolist())
    allocated = _allocate_counts(global_counts, num_trajectories)
    selected: list[int] = []
    for collector_id in sorted(allocated):
        candidates = (modes == collector_id).nonzero(as_tuple=False).flatten()
        count = int(allocated[collector_id])
        if candidates.numel() == 0 or count <= 0:
            continue
        permutation = torch.randperm(candidates.numel(), generator=generator)
        selected.extend(candidates[permutation[:count]].tolist())
    if len(selected) < num_trajectories:
        mask = torch.ones(num_envs, dtype=torch.bool)
        if selected:
            mask[torch.tensor(selected, dtype=torch.long)] = False
        remaining = mask.nonzero(as_tuple=False).flatten()
        permutation = torch.randperm(remaining.numel(), generator=generator)
        selected.extend(
            remaining[permutation[: num_trajectories - len(selected)]].tolist()
        )
    result = torch.tensor(selected[:num_trajectories], dtype=torch.long).sort().values
    if result.numel() != num_trajectories or torch.unique(result).numel() != result.numel():
        raise RuntimeError("Stratified selection did not produce unique environment IDs.")
    return result


def _select_env_rows(value: Any, env_ids: torch.Tensor, num_source_envs: int) -> Any:
    """Recursively select an environment dimension at position zero."""

    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == num_source_envs:
            return value.index_select(0, env_ids).clone()
        return value.clone()
    if isinstance(value, dict):
        return {
            key: _select_env_rows(item, env_ids, num_source_envs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_select_env_rows(item, env_ids, num_source_envs) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _select_env_rows(item, env_ids, num_source_envs) for item in value
        )
    return copy.deepcopy(value)


def _slice_dataset(
    dataset: dict[str, Any],
    env_ids: torch.Tensor,
    *,
    num_time_steps: int,
    num_source_envs: int,
) -> tuple[dict[str, Any], list[str]]:
    sliced: dict[str, Any] = {}
    time_major_keys: list[str] = []
    for key, value in dataset.items():
        if isinstance(value, list) and len(value) == num_time_steps:
            sliced[key] = [
                _select_env_rows(row, env_ids, num_source_envs) for row in value
            ]
            time_major_keys.append(key)
        elif (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and int(value.shape[0]) == num_time_steps
            and int(value.shape[1]) == num_source_envs
        ):
            sliced[key] = value.index_select(1, env_ids).clone()
            time_major_keys.append(key)
        elif key == "metadata":
            sliced[key] = copy.deepcopy(value or {})
        elif key not in ("startup_domain_table", "episode_domain_table"):
            sliced[key] = copy.deepcopy(value)

    startup = dataset.get("startup_domain_table")
    if isinstance(startup, dict):
        filtered = {}
        for key, value in startup.items():
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == num_source_envs:
                filtered[key] = value.index_select(0, env_ids).clone()
            else:
                filtered[key] = copy.deepcopy(value)
        if "startup_domain_id" in filtered:
            filtered["startup_domain_id"] = torch.arange(env_ids.numel(), dtype=torch.long)
        sliced["startup_domain_table"] = filtered

    episode_table = dataset.get("episode_domain_table")
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
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == len(source_ids):
                filtered[key] = value[keep].clone()
            else:
                filtered[key] = copy.deepcopy(value)
        filtered["startup_domain_id"] = remapped[keep]
        sliced["episode_domain_table"] = filtered

    return sliced, sorted(time_major_keys)


def _collector_statistics(dataset: dict[str, Any]) -> dict[str, Any]:
    collector = _stack_key(dataset, "collector_types").long()
    flat_counts = torch.bincount(
        collector.reshape(-1),
        minlength=max(COLLECTOR_ID_TO_NAME) + 1,
    )
    total = int(collector.numel())
    transition_counts = {
        name: int(flat_counts[collector_id])
        for collector_id, name in sorted(COLLECTOR_ID_TO_NAME.items())
    }
    transition_ratios = {
        name: count / total for name, count in transition_counts.items()
    }
    modes = _env_collector_modes(dataset)
    mode_counter = Counter(int(value) for value in modes.tolist())
    mode_counts = {
        COLLECTOR_ID_TO_NAME.get(collector_id, str(collector_id)): int(count)
        for collector_id, count in sorted(mode_counter.items())
    }
    return {
        "transition_counts": transition_counts,
        "transition_ratios": transition_ratios,
        "environment_mode_counts": mode_counts,
    }


def _validate_selected_dataset(
    dataset: dict[str, Any],
    *,
    expected_time_steps: int,
    expected_num_envs: int,
    require_trace_snapshots: bool,
) -> dict[str, Any]:
    expected_prefix = (expected_time_steps, expected_num_envs)
    core_widths = {
        "states": 45,
        "next_states": 45,
        "actions": 12,
        "observations": 45,
        "commands": 3,
        "prev_actions": 12,
    }
    for key, width in core_widths.items():
        tensor = _stack_key(dataset, key)
        if tuple(tensor.shape) != (*expected_prefix, width):
            raise ValueError(
                f"{key} shape {tuple(tensor.shape)} != {(*expected_prefix, width)}"
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains NaN/Inf.")

    for key in ("dones", "timeouts", "episode_ids", "timesteps", "collector_types"):
        tensor = _stack_key(dataset, key)
        if tuple(tensor.shape[:2]) != expected_prefix:
            raise ValueError(f"{key} prefix {tuple(tensor.shape[:2])} != {expected_prefix}")

    if require_trace_snapshots:
        for key, width in V13_SNAPSHOT_WIDTHS.items():
            tensor = _stack_key(dataset, key)
            if tuple(tensor.shape) != (*expected_prefix, width):
                raise ValueError(
                    f"{key} shape {tuple(tensor.shape)} != {(*expected_prefix, width)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{key} contains NaN/Inf.")
        trace_valid = _stack_key(dataset, "trace_valid_masks").bool()
        if tuple(trace_valid.shape[:2]) != expected_prefix or not trace_valid.all():
            raise ValueError("trace_valid_masks must be true for every selected transition.")

    episode_ids = _stack_key(dataset, "episode_ids").long()
    episode_changes = int((episode_ids[1:] != episode_ids[:-1]).sum().item())
    dones = _stack_key(dataset, "dones").bool()
    return {
        **_collector_statistics(dataset),
        "unique_episode_ids": int(torch.unique(episode_ids).numel()),
        "episode_changes": episode_changes,
        "done_count": int(dones.sum().item()),
    }


def main() -> None:
    args = _parse_args()
    source_path = Path(args.source_path).expanduser().resolve()
    save_path = Path(args.save_path).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    launch_manifest_path = Path(args.launch_manifest_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not launch_manifest_path.is_file():
        raise FileNotFoundError(launch_manifest_path)
    if save_path.exists():
        raise FileExistsError(f"Refusing to overwrite selected dataset: {save_path}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite selection report: {report_path}")

    dataset = load_mixed_dataset(source_path)
    _validate_requested_mix(dataset)
    _validate_source_protocol(
        dataset,
        condition_id=str(args.condition_id),
        expected_expert_policy_path=str(args.expected_expert_policy_path),
    )
    launch_manifest = json.loads(launch_manifest_path.read_text(encoding="utf-8"))
    if launch_manifest.get("schema") != "go2_v13_mixed_collection_launch_v1":
        raise ValueError(
            f"Unexpected launch manifest schema: {launch_manifest.get('schema')!r}"
        )
    if str(launch_manifest.get("resolved_condition", {}).get("condition_id")) != str(
        args.condition_id
    ):
        raise ValueError("Launch manifest condition_id does not match selection condition.")
    manifest_raw = str(
        Path(launch_manifest.get("paths", {}).get("raw_dataset", "")).expanduser().resolve()
    )
    if manifest_raw != str(source_path):
        raise ValueError(
            f"Launch manifest raw dataset is {manifest_raw!r}, expected {str(source_path)!r}."
        )
    manifest_actor = str(
        Path(
            launch_manifest.get("artifacts", {}).get("expert_actor", {}).get("path", "")
        ).expanduser().resolve()
    )
    expected_actor = str(
        (Path(args.expected_expert_policy_path).expanduser().resolve() / "actor.pt")
    )
    if manifest_actor != expected_actor:
        raise ValueError(
            f"Launch manifest expert actor is {manifest_actor!r}, expected {expected_actor!r}."
        )
    launch_manifest_sha256 = _sha256(launch_manifest_path)
    states = _stack_key(dataset, "states")
    if states.ndim != 3 or int(states.shape[-1]) != 45:
        raise ValueError(f"states must have shape [T,E,45], got {tuple(states.shape)}")
    num_time_steps, num_source_envs = (int(states.shape[0]), int(states.shape[1]))
    if num_time_steps != int(args.expected_time_steps):
        raise ValueError(
            f"Expected {args.expected_time_steps} source time steps, got {num_time_steps}."
        )
    if int(dataset.get("num_envs", -1)) != num_source_envs:
        raise ValueError(
            f"dataset.num_envs={dataset.get('num_envs')} but states has {num_source_envs}."
        )

    env_ids = _select_env_ids(
        dataset,
        int(args.num_trajectories),
        int(args.seed),
        str(args.selection),
    )
    sliced, time_major_keys = _slice_dataset(
        dataset,
        env_ids,
        num_time_steps=num_time_steps,
        num_source_envs=num_source_envs,
    )
    num_selected = int(env_ids.numel())
    source_sha256 = _sha256(source_path)
    metadata = dict(sliced.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": str(source_path),
            "source_dataset_sha256": source_sha256,
            "launch_manifest_path": str(launch_manifest_path),
            "launch_manifest_sha256": launch_manifest_sha256,
            "subset_kind": "v13_mixed_environment_columns",
            "condition_id": str(args.condition_id),
            "selection": str(args.selection),
            "selection_seed": int(args.seed),
            "selected_env_ids": env_ids.tolist(),
            "num_trajectories": num_selected,
            "num_time_steps": num_time_steps,
            "num_transitions": num_time_steps * num_selected,
            "actual_num_transitions": num_time_steps * num_selected,
            "time_major_keys_selected": time_major_keys,
            "trace_snapshots_required": bool(args.require_trace_snapshots),
        }
    )
    sliced["metadata"] = metadata
    sliced["num_envs"] = num_selected
    sliced["capacity"] = num_time_steps * num_selected
    sliced["source_env_ids"] = [
        env_ids.clone() for _ in range(num_time_steps)
    ]

    statistics = _validate_selected_dataset(
        sliced,
        expected_time_steps=num_time_steps,
        expected_num_envs=num_selected,
        require_trace_snapshots=bool(args.require_trace_snapshots),
    )
    sliced["metadata"].update(
        {
            "actual_collector_transition_counts": statistics["transition_counts"],
            "actual_collector_transition_ratios": statistics["transition_ratios"],
            "collector_environment_mode_counts": statistics["environment_mode_counts"],
        }
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_dict(sliced, save_path)
    output_sha256 = _sha256(save_path)
    report = {
        "schema": "go2_v13_mixed_stratified_selection_v1",
        "condition_id": str(args.condition_id),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "launch_manifest_path": str(launch_manifest_path),
        "launch_manifest_sha256": launch_manifest_sha256,
        "output_path": str(save_path),
        "output_sha256": output_sha256,
        "selection": str(args.selection),
        "selection_seed": int(args.seed),
        "selected_env_ids": env_ids.tolist(),
        "num_time_steps": num_time_steps,
        "num_envs": num_selected,
        "num_transitions": num_time_steps * num_selected,
        "requested_collector_mix": V13_COLLECTOR_MIX,
        "time_major_keys_selected": time_major_keys,
        **statistics,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[V13-MixedSelection] source={source_path}")
    print(f"[V13-MixedSelection] output={save_path}")
    print(f"[V13-MixedSelection] report={report_path}")
    print(f"[V13-MixedSelection] selected_env_ids={env_ids.tolist()}")
    print(
        "[V13-MixedSelection] collector_transition_ratios="
        f"{json.dumps(statistics['transition_ratios'], sort_keys=True)}"
    )
    print(f"[V13-MixedSelection] sha256={output_sha256}")


if __name__ == "__main__":
    main()
