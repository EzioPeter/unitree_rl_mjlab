#!/usr/bin/env python3
"""Certify Go2 V1 manual reset against a simulator-generated source buffer.

This module deliberately contains no MJLab imports.  It owns the CPU-only
dataset preflight, deterministic window selection, expected-value assembly,
error aggregation, and PASS/FAIL/BLOCKED artifact writing.  A Phase-B
environment implementation can be supplied with ``--runner module:callable``
or by calling :func:`run_reset_certification` with a Python callback.

The callback receives a :class:`ResetValidationRequest` and a JSON-compatible
runner configuration.  It must return :class:`ResetRunnerOutput` or a mapping
with the same fields.  Snapshot action histories are copied verbatim from the
V1 simulator fields; they are never overwritten with ``prev_actions``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from scripts.reinforcement_learning.rwm_trace.condition_registry import (
    ConditionRegistryError,
    assert_condition_metadata,
)
from scripts.reinforcement_learning.rwm_trace.device_guard import DeviceGuardError


FORMAT_VERSION = "go2_trace_reset_certification_v1"
SNAPSHOT_VERSION = "go2_trace_snapshot_v1"
DEFAULT_STATE_TOLERANCE = 1.0e-4
DEFAULT_HISTORY_COMMAND_TOLERANCE = 1.0e-7
MAX_STATE_TOLERANCE = 1.0e-4
MAX_HISTORY_COMMAND_TOLERANCE = 1.0e-7
SOURCE_CONTINUITY_TOLERANCE = 1.0e-7
FORMAL_RUNNER_SPEC = (
    "scripts.reinforcement_learning.rwm_trace.go2_perfect_sim_runner:"
    "run_perfect_simulator"
)


class CertificationBlockedError(RuntimeError):
    """An external runtime/dependency/device condition prevented certification."""

V1_SNAPSHOT_FIELDS: dict[str, tuple[str, int]] = {
    "root_state_local": ("sim_root_states_local", 13),
    "joint_position": ("sim_joint_positions", 12),
    "joint_velocity": ("sim_joint_velocities", 12),
    "action": ("sim_action_histories", 12),
    "prev_action": ("sim_prev_action_histories", 12),
    "prev_prev_action": ("sim_prev_prev_action_histories", 12),
    "command": ("sim_snapshot_commands", 3),
}

REQUIRED_DATASET_FIELDS: dict[str, int] = {
    "states": 45,
    "next_states": 45,
    "actions": 12,
    "commands": 3,
    "contacts": 4,
    "terminations": 1,
    "episode_ids": 1,
    "timesteps": 1,
    **{dataset_key: width for dataset_key, width in V1_SNAPSHOT_FIELDS.values()},
}

OPTIONAL_DATASET_FIELDS: dict[str, int] = {
    "dones": 1,
    "timeouts": 1,
    "rewards": 1,
    "prev_actions": 12,
    "raw_actions": 12,
}

PHYSICAL37_BLOCKS: dict[str, tuple[int, int]] = {
    "root_state_local": (0, 13),
    "joint_position": (13, 25),
    "joint_velocity": (25, 37),
}

RWM45_BLOCKS: dict[str, tuple[int, int]] = {
    "base_lin_vel": (0, 3),
    "base_ang_vel": (3, 6),
    "projected_gravity": (6, 9),
    "joint_position_relative": (9, 21),
    "joint_velocity": (21, 33),
    "actuator_force": (33, 45),
}


class ResetRunner(Protocol):
    """Adapter contract for a real perfect-simulator runner."""

    def __call__(
        self,
        request: "ResetValidationRequest",
        runner_config: Mapping[str, Any],
    ) -> "ResetRunnerOutput | Mapping[str, Any]": ...


@dataclass(frozen=True)
class SourceWindow:
    """One H-transition source-buffer window with H+1 state points."""

    env_id: int
    start_time_index: int
    point_time_indices: tuple[int, ...]
    episode_id: int
    start_timestep: int
    num_envs: int

    @property
    def transition_time_indices(self) -> tuple[int, ...]:
        return self.point_time_indices[:-1]

    @property
    def start_flat_row(self) -> int:
        return self.start_time_index * self.num_envs + self.env_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "start_time_index": self.start_time_index,
            "start_flat_row": self.start_flat_row,
            "point_time_indices": list(self.point_time_indices),
            "episode_id": self.episode_id,
            "start_timestep": self.start_timestep,
        }


@dataclass
class DatasetView:
    """Validated dataset fields normalized to [time, env, width]."""

    fields: dict[str, torch.Tensor]
    time_steps: int
    num_envs: int
    preflight: dict[str, Any]


@dataclass
class ResetValidationRequest:
    """Batched values passed to a perfect-simulator runner."""

    snapshot: dict[str, torch.Tensor | str]
    actions: torch.Tensor
    transition_commands: torch.Tensor
    expected_physical_states: torch.Tensor
    expected_rwm_states: torch.Tensor
    expected_action_histories: torch.Tensor
    expected_prev_action_histories: torch.Tensor
    expected_prev_prev_action_histories: torch.Tensor
    expected_commands: torch.Tensor
    expected_contacts: torch.Tensor
    expected_terminations: torch.Tensor
    expected_rewards: torch.Tensor | None
    windows: tuple[SourceWindow, ...]
    horizon: int


@dataclass
class ResetRunnerOutput:
    """Observed perfect-simulator rollout, including the reset point."""

    physical_states: torch.Tensor
    rwm_states: torch.Tensor
    action_histories: torch.Tensor
    prev_action_histories: torch.Tensor
    prev_prev_action_histories: torch.Tensor
    commands: torch.Tensor
    contacts: torch.Tensor
    terminations: torch.Tensor
    rewards: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_binary_tensor(value: torch.Tensor, *, name: str) -> None:
    """Require a boolean tensor or a numeric tensor containing only 0/1."""

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.dtype == torch.bool:
        return
    if tensor.is_complex() or not (
        torch.is_floating_point(tensor)
        or tensor.dtype
        in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    ):
        raise ValueError(f"{name} must be bool or a numeric 0/1 tensor, got {tensor.dtype}.")
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{name} contains non-finite values.")
    if not bool(((tensor == 0) | (tensor == 1)).all()):
        raise ValueError(f"{name} must contain only exact binary values 0 or 1.")


def _runner_exception_status(exc: BaseException) -> str:
    """Classify only external availability failures as BLOCKED."""

    blocked_types = (
        CertificationBlockedError,
        DeviceGuardError,
        ImportError,
        OSError,
    )
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, blocked_types):
            return "BLOCKED"
        current = current.__cause__ or current.__context__

    # CUDA runtime failures are commonly surfaced as an untyped RuntimeError.
    # Keep this list narrow so simulator/config/numerical RuntimeErrors remain FAIL.
    message = str(exc).lower()
    cuda_unavailable_markers = (
        "cuda driver",
        "cuda is not available",
        "no cuda-capable device",
        "no cuda devices",
        "nvidia-smi",
        "failed to initialize nvml",
        "cuda out of memory",
        "device is busy or unavailable",
    )
    if isinstance(exc, RuntimeError) and any(
        marker in message for marker in cuda_unavailable_markers
    ):
        return "BLOCKED"
    return "FAIL"


def _dataset_runner_metadata_binding(
    dataset: Mapping[str, Any],
    runner_config: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any]:
    """Bind source collection metadata to the exact perfect-runner config."""

    if not required:
        return {
            "required": False,
            "passed": True,
            "reason": "test-only callbacks cannot produce a formal certificate",
        }

    metadata = dataset.get("metadata")
    if not isinstance(metadata, Mapping):
        return {
            "required": True,
            "passed": False,
            "issues": ["dataset metadata mapping is required"],
            "comparisons": {},
        }

    issues: list[str] = []
    comparisons: dict[str, Any] = {}
    for key in ("physics", "provenance", "device_identity"):
        source_value = metadata.get(key)
        runner_value = runner_config.get(key)
        source_is_mapping = isinstance(source_value, Mapping)
        runner_is_mapping = isinstance(runner_value, Mapping)
        try:
            source_hash = (
                _canonical_sha256(source_value) if source_is_mapping else None
            )
        except (TypeError, ValueError):
            source_hash = None
            source_is_mapping = False
        try:
            runner_hash = (
                _canonical_sha256(runner_value) if runner_is_mapping else None
            )
        except (TypeError, ValueError):
            runner_hash = None
            runner_is_mapping = False
        matched = bool(
            source_is_mapping
            and runner_is_mapping
            and source_hash == runner_hash
        )
        comparisons[key] = {
            "matched": matched,
            "dataset_sha256": source_hash,
            "runner_config_sha256": runner_hash,
        }
        if not matched:
            issues.append(
                f"dataset metadata {key!r} does not exactly match runner_config"
            )

    dataset_seed = metadata.get("seed")
    runner_seed = runner_config.get("seed")
    seed_valid = bool(
        not isinstance(dataset_seed, bool)
        and isinstance(dataset_seed, int)
        and not isinstance(runner_seed, bool)
        and isinstance(runner_seed, int)
    )
    seed_matched = bool(seed_valid and dataset_seed == runner_seed)
    comparisons["seed"] = {
        "matched": seed_matched,
        "dataset": dataset_seed,
        "runner_config": runner_seed,
    }
    if not seed_matched:
        issues.append(
            "dataset metadata 'seed' does not exactly match runner_config"
        )

    dataset_burn_in = metadata.get("history_burn_in_steps")
    runner_burn_in = runner_config.get("history_burn_in_steps")
    burn_in_valid = bool(
        not isinstance(dataset_burn_in, bool)
        and isinstance(dataset_burn_in, int)
        and dataset_burn_in >= 3
        and not isinstance(runner_burn_in, bool)
        and isinstance(runner_burn_in, int)
        and runner_burn_in >= 3
    )
    burn_in_matched = bool(
        burn_in_valid and dataset_burn_in == runner_burn_in
    )
    comparisons["history_burn_in_steps"] = {
        "matched": burn_in_matched,
        "dataset": dataset_burn_in,
        "runner_config": runner_burn_in,
        "minimum_required": 3,
    }
    if not burn_in_matched:
        issues.append(
            "dataset metadata 'history_burn_in_steps' must exactly match "
            "runner_config and be at least 3"
        )

    return {
        "required": True,
        "passed": not issues,
        "issues": issues,
        "comparisons": comparisons,
    }


def implementation_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = {
        "validator": Path(__file__).resolve(),
        "manual_reset": root / "simulator_reset.py",
        "condition_registry": root / "condition_registry.py",
        "device_guard": root / "device_guard.py",
        "perfect_runner": root / "go2_perfect_sim_runner.py",
    }
    return {
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path) if path.is_file() else None,
            }
            for name, path in files.items()
        },
        "python": sys.version,
        "torch": torch.__version__,
    }


def _stack_raw_field(dataset: Mapping[str, Any], key: str) -> torch.Tensor:
    raw = dataset[key]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError(f"Dataset field {key!r} is empty.")
        return torch.stack([torch.as_tensor(value) for value in raw], dim=0)
    return torch.as_tensor(raw)


def _num_envs_hint(dataset: Mapping[str, Any]) -> int | None:
    values = [
        dataset.get("num_envs"),
        (dataset.get("metadata") or {}).get("num_envs")
        if isinstance(dataset.get("metadata"), Mapping)
        else None,
    ]
    for value in values:
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _normalize_time_env_width(
    value: torch.Tensor,
    *,
    key: str,
    width: int,
    num_envs_hint: int | None,
) -> torch.Tensor:
    """Normalize common list/tensor dataset encodings to [T,N,D]."""

    value = value.detach().cpu()
    if value.ndim == 0:
        if width != 1:
            raise ValueError(f"{key} scalar cannot represent width {width}.")
        value = value.reshape(1, 1, 1)
    elif value.ndim == 1:
        if width > 1:
            if int(value.numel()) != width:
                raise ValueError(f"{key} has shape {tuple(value.shape)}, expected width {width}.")
            value = value.reshape(1, 1, width)
        elif num_envs_hint and value.numel() % num_envs_hint == 0:
            value = value.reshape(-1, num_envs_hint, 1)
        else:
            value = value.reshape(-1, 1, 1)
    elif value.ndim == 2:
        if width > 1:
            if int(value.shape[-1]) != width:
                raise ValueError(f"{key} has shape {tuple(value.shape)}, expected trailing width {width}.")
            if num_envs_hint and num_envs_hint > 1 and value.shape[0] % num_envs_hint == 0:
                value = value.reshape(-1, num_envs_hint, width)
            else:
                value = value.unsqueeze(1)
        elif int(value.shape[-1]) == 1:
            if num_envs_hint and num_envs_hint > 1 and value.shape[0] % num_envs_hint == 0:
                value = value.reshape(-1, num_envs_hint, 1)
            else:
                value = value.unsqueeze(1)
        else:
            value = value.unsqueeze(-1)
    elif value.ndim == 3:
        if int(value.shape[-1]) != width:
            raise ValueError(f"{key} has shape {tuple(value.shape)}, expected trailing width {width}.")
    else:
        raise ValueError(f"{key} has unsupported shape {tuple(value.shape)}.")
    if value.ndim != 3 or int(value.shape[-1]) != width:
        raise ValueError(f"{key} normalized to invalid shape {tuple(value.shape)}.")
    return value


def preflight_dataset(
    dataset: Mapping[str, Any],
    *,
    expected_condition_id: str | None = None,
    expected_task: str | None = None,
    expected_horizon: int | None = None,
    expected_runtime_scope: str | None = None,
) -> tuple[DatasetView | None, dict[str, Any]]:
    """Validate V1 fields without importing or constructing a simulator."""

    issues: list[str] = []
    warnings: list[str] = []
    missing = sorted(key for key in REQUIRED_DATASET_FIELDS if key not in dataset)
    report: dict[str, Any] = {
        "passed": False,
        "snapshot_version": SNAPSHOT_VERSION,
        "required_snapshot_fields": {
            target: {"dataset_field": source, "width": width}
            for target, (source, width) in V1_SNAPSHOT_FIELDS.items()
        },
        "missing_required_fields": missing,
        "issues": issues,
        "warnings": warnings,
    }
    if missing:
        issues.append("missing required dataset fields: " + ", ".join(missing))
        return None, report

    metadata = dataset.get("metadata")
    if not isinstance(metadata, Mapping):
        issues.append("dataset metadata mapping is required")
        metadata = {}
    if metadata.get("snapshot_version") != SNAPSHOT_VERSION:
        issues.append(
            "metadata.snapshot_version must be "
            f"{SNAPSHOT_VERSION!r}, got {metadata.get('snapshot_version')!r}"
        )
    condition_id = metadata.get("condition_id")
    task = metadata.get("task")
    metadata_horizon = metadata.get("horizon")
    runtime_scope = metadata.get("runtime_scope")
    if not isinstance(condition_id, str) or not condition_id:
        issues.append("metadata.condition_id is required")
    if not isinstance(task, str) or not task:
        issues.append("metadata.task is required")
    if (
        isinstance(metadata_horizon, bool)
        or not isinstance(metadata_horizon, int)
        or metadata_horizon < 1
    ):
        issues.append("metadata.horizon must be a positive integer")
    if not isinstance(runtime_scope, str) or not runtime_scope:
        issues.append("metadata.runtime_scope is required")
    if expected_condition_id is not None and condition_id != expected_condition_id:
        issues.append(
            f"dataset condition {condition_id!r} does not match "
            f"{expected_condition_id!r}"
        )
    if expected_task is not None and task != expected_task:
        issues.append(f"dataset task {task!r} does not match {expected_task!r}")
    if expected_horizon is not None and metadata_horizon != int(expected_horizon):
        issues.append(
            f"dataset horizon {metadata_horizon!r} does not match "
            f"{int(expected_horizon)!r}"
        )
    if (
        expected_runtime_scope is not None
        and runtime_scope != expected_runtime_scope
    ):
        issues.append(
            f"dataset runtime_scope {runtime_scope!r} does not match "
            f"{expected_runtime_scope!r}"
        )
    condition_metadata = metadata.get("condition_registry")
    try:
        assert_condition_metadata(
            condition_metadata,
            expected_condition_id=(
                expected_condition_id
                if expected_condition_id is not None
                else condition_id if isinstance(condition_id, str) else None
            ),
        )
    except (ConditionRegistryError, TypeError) as exc:
        issues.append(f"invalid condition registry metadata: {exc}")

    controls = metadata.get("certification_controls")
    required_controls = {
        "action_delay": 0,
        "actuator_delay": 0,
        "domain_randomization": False,
        "push_randomization": False,
        "observation_noise": False,
        "action_noise": False,
        "physics_startup_events": False,
        "physics_interval_events": False,
        "command_random_resample": False,
    }
    if not isinstance(controls, Mapping):
        issues.append("metadata.certification_controls mapping is required")
    else:
        for key, expected in required_controls.items():
            if controls.get(key) != expected:
                issues.append(
                    f"certification control {key!r} must be {expected!r}, "
                    f"got {controls.get(key)!r}"
                )

    hint = _num_envs_hint(dataset)
    fields: dict[str, torch.Tensor] = {}
    for key, width in {**REQUIRED_DATASET_FIELDS, **OPTIONAL_DATASET_FIELDS}.items():
        if key not in dataset:
            continue
        try:
            fields[key] = _normalize_time_env_width(
                _stack_raw_field(dataset, key),
                key=key,
                width=width,
                num_envs_hint=hint,
            )
        except Exception as exc:
            issues.append(f"{key}: {exc}")
    if issues:
        return None, report

    reference_shape = tuple(fields["states"].shape[:2])
    mismatched_shapes = {
        key: tuple(value.shape)
        for key, value in fields.items()
        if tuple(value.shape[:2]) != reference_shape
    }
    if mismatched_shapes:
        issues.append(
            f"dataset time/env shape mismatch; expected {reference_shape}: {mismatched_shapes}"
        )
        return None, report

    floating_fields = {
        "states",
        "next_states",
        "actions",
        "commands",
        "rewards",
        "prev_actions",
        "raw_actions",
        *(source for source, _width in V1_SNAPSHOT_FIELDS.values()),
    }
    wrong_dtypes = {
        key: str(value.dtype)
        for key, value in fields.items()
        if key in floating_fields and not torch.is_floating_point(value)
    }
    if wrong_dtypes:
        issues.append(f"floating dataset fields have invalid dtypes: {wrong_dtypes}")

    for binary_name in ("contacts", "terminations", "dones", "timeouts"):
        if binary_name not in fields:
            continue
        try:
            _require_binary_tensor(fields[binary_name], name=binary_name)
        except ValueError as exc:
            issues.append(str(exc))

    nonfinite_fields = []
    for key, value in fields.items():
        if key in {"episode_ids", "timesteps", "dones", "timeouts", "terminations"}:
            continue
        if not bool(torch.isfinite(value.float()).all()):
            if key == "rewards":
                warnings.append("rewards contains nonfinite values; reward diagnostics may be unavailable")
            else:
                nonfinite_fields.append(key)
    if nonfinite_fields:
        issues.append("nonfinite required values in: " + ", ".join(sorted(nonfinite_fields)))

    episode_ids = fields["episode_ids"].long()
    timesteps = fields["timesteps"].long()
    if bool((timesteps < 0).any()):
        issues.append("timesteps contains negative values")
    duplicate_keys = 0
    for env_id in range(reference_shape[1]):
        keys = list(
            zip(
                episode_ids[:, env_id, 0].tolist(),
                timesteps[:, env_id, 0].tolist(),
            )
        )
        duplicate_keys += len(keys) - len(set(keys))
    if duplicate_keys:
        issues.append(f"found {duplicate_keys} duplicate (episode_id,timestep) keys within env streams")

    declared_terminal = fields["terminations"].bool().squeeze(-1)
    if "timeouts" in fields:
        declared_terminal = declared_terminal | fields["timeouts"].bool().squeeze(-1)
    done_consistency_mismatch = 0
    if "dones" in fields:
        done_values = fields["dones"].bool().squeeze(-1)
        done_consistency_mismatch = int((done_values != declared_terminal).sum())
        if done_consistency_mismatch:
            issues.append(
                "dones disagrees with terminations|timeouts at "
                f"{done_consistency_mismatch} rows"
            )

    source_action_history_disagreement = None
    if "prev_actions" in fields:
        source_action_history_disagreement = float(
            torch.max(
                torch.abs(
                    fields["sim_action_histories"].float()
                    - fields["prev_actions"].float()
                )
            )
        )
        if source_action_history_disagreement > SOURCE_CONTINUITY_TOLERANCE:
            issues.append(
                "sim_action_histories differs from row-capture prev_actions: "
                f"L∞={source_action_history_disagreement}"
            )

    episodes = fields["episode_ids"].long().squeeze(-1)
    steps = fields["timesteps"].long().squeeze(-1)
    adjacent = (
        (episodes[:-1] == episodes[1:])
        & (steps[1:] == steps[:-1] + 1)
    )
    continuity_error = torch.abs(
        fields["next_states"][:-1].float() - fields["states"][1:].float()
    )
    if bool(adjacent.any()):
        source_continuity_linf = float(continuity_error[adjacent].max())
        source_continuity_count = int(adjacent.sum())
        if source_continuity_linf > SOURCE_CONTINUITY_TOLERANCE:
            issues.append(
                "next_states[t] disagrees with states[t+1] on continuous rows: "
                f"L∞={source_continuity_linf}"
            )
    else:
        source_continuity_linf = None
        source_continuity_count = 0

    history_adjacent = adjacent & ~declared_terminal[:-1]
    action_history_contract = {
        "action_from_executed_action": (
            fields["sim_action_histories"][1:].float()
            - fields["actions"][:-1].float()
        ).abs(),
        "prev_action_from_action": (
            fields["sim_prev_action_histories"][1:].float()
            - fields["sim_action_histories"][:-1].float()
        ).abs(),
        "prev_prev_action_from_prev_action": (
            fields["sim_prev_prev_action_histories"][1:].float()
            - fields["sim_prev_action_histories"][:-1].float()
        ).abs(),
    }
    action_history_contract_linf: dict[str, float | None] = {}
    history_contract_pair_count = int(history_adjacent.sum())
    for contract_name, error in action_history_contract.items():
        contract_linf = (
            float(error[history_adjacent].max())
            if history_contract_pair_count
            else None
        )
        action_history_contract_linf[contract_name] = contract_linf
        if (
            contract_linf is not None
            and contract_linf > SOURCE_CONTINUITY_TOLERANCE
        ):
            issues.append(
                f"action-history contract {contract_name} failed on adjacent "
                f"nonterminal rows: L∞={contract_linf}"
            )

    raw_action_linf = None
    if "raw_actions" in fields:
        raw_action_linf = float(
            torch.max(torch.abs(fields["raw_actions"].float() - fields["actions"].float()))
        )
        if (
            raw_action_linf > SOURCE_CONTINUITY_TOLERANCE
            and not metadata.get("raw_to_executed_action_transform")
        ):
            issues.append(
                "raw_actions differs from actions but metadata lacks "
                "raw_to_executed_action_transform"
            )

    report.update(
        {
            "passed": not issues,
            "time_steps": int(reference_shape[0]),
            "num_envs": int(reference_shape[1]),
            "field_shapes": {key: list(value.shape) for key, value in sorted(fields.items())},
            "source_action_history_vs_prev_actions_linf": source_action_history_disagreement,
            "source_next_state_continuity_linf": source_continuity_linf,
            "source_next_state_continuity_pair_count": source_continuity_count,
            "action_history_contract_linf": action_history_contract_linf,
            "action_history_contract_pair_count": history_contract_pair_count,
            "done_consistency_mismatch_count": done_consistency_mismatch,
            "raw_actions_vs_actions_linf": raw_action_linf,
            "metadata_condition_id": condition_id,
            "metadata_task": task,
            "metadata_horizon": metadata_horizon,
            "metadata_runtime_scope": runtime_scope,
            "canonical_metadata_sha256": _canonical_sha256(metadata),
        }
    )
    if issues:
        return None, report
    return DatasetView(
        fields=fields,
        time_steps=int(reference_shape[0]),
        num_envs=int(reference_shape[1]),
        preflight=report,
    ), report


def _transition_terminal_mask(view: DatasetView) -> torch.Tensor:
    fields = view.fields
    terminal = fields["terminations"].bool().squeeze(-1)
    if "dones" in fields:
        terminal |= fields["dones"].bool().squeeze(-1)
    if "timeouts" in fields:
        terminal |= fields["timeouts"].bool().squeeze(-1)
    return terminal


def select_source_windows(
    view: DatasetView,
    *,
    horizon: int,
    count: int,
    seed: int,
    min_source_timestep: int = 0,
) -> tuple[list[SourceWindow], dict[str, Any]]:
    """Select deterministic H-transition windows with H+1 buffer points."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if count < 1:
        raise ValueError("count must be >= 1")
    episodes = view.fields["episode_ids"].long().squeeze(-1)
    timesteps = view.fields["timesteps"].long().squeeze(-1)
    terminal = _transition_terminal_mask(view)
    candidates: list[SourceWindow] = []
    rejection_counts = {
        "insufficient_h_plus_one_rows": 0,
        "below_min_timestep": 0,
        "episode_boundary": 0,
        "nonconsecutive_timestep": 0,
        "terminal_transition": 0,
    }
    for env_id in range(view.num_envs):
        for start in range(view.time_steps):
            stop_point = start + horizon
            if stop_point >= view.time_steps:
                rejection_counts["insufficient_h_plus_one_rows"] += 1
                continue
            start_timestep = int(timesteps[start, env_id])
            if start_timestep < min_source_timestep:
                rejection_counts["below_min_timestep"] += 1
                continue
            point_slice = slice(start, stop_point + 1)
            episode_values = episodes[point_slice, env_id]
            if not bool(torch.all(episode_values == episode_values[0])):
                rejection_counts["episode_boundary"] += 1
                continue
            expected_timesteps = torch.arange(
                start_timestep,
                start_timestep + horizon + 1,
                dtype=timesteps.dtype,
            )
            if not torch.equal(timesteps[point_slice, env_id], expected_timesteps):
                rejection_counts["nonconsecutive_timestep"] += 1
                continue
            if bool(terminal[start:stop_point, env_id].any()):
                rejection_counts["terminal_transition"] += 1
                continue
            candidates.append(
                SourceWindow(
                    env_id=env_id,
                    start_time_index=start,
                    point_time_indices=tuple(range(start, stop_point + 1)),
                    episode_id=int(episode_values[0]),
                    start_timestep=start_timestep,
                    num_envs=view.num_envs,
                )
            )
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    selected = candidates[:count]
    report = {
        "horizon": int(horizon),
        "required_point_count_per_window": int(horizon + 1),
        "requested_window_count": int(count),
        "available_window_count": int(len(candidates)),
        "selected_window_count": int(len(selected)),
        "seed": int(seed),
        "min_source_timestep": int(min_source_timestep),
        "all_h_transitions_must_be_nonterminal": True,
        "rejection_counts": rejection_counts,
        "selected_windows": [window.to_dict() for window in selected],
        "passed": len(selected) == count,
    }
    if len(selected) != count:
        report["error"] = (
            f"Only {len(candidates)} valid H+1 windows are available; {count} requested."
        )
    return selected, report


def _gather_points(
    value: torch.Tensor,
    windows: Sequence[SourceWindow],
) -> torch.Tensor:
    return torch.stack(
        [
            value[list(window.point_time_indices), window.env_id]
            for window in windows
        ],
        dim=0,
    )


def _gather_transitions(
    value: torch.Tensor,
    windows: Sequence[SourceWindow],
) -> torch.Tensor:
    return torch.stack(
        [
            value[list(window.transition_time_indices), window.env_id]
            for window in windows
        ],
        dim=0,
    )


def prepare_validation_request(
    view: DatasetView,
    windows: Sequence[SourceWindow],
    *,
    command_source_tolerance: float = DEFAULT_HISTORY_COMMAND_TOLERANCE,
) -> tuple[ResetValidationRequest, dict[str, Any]]:
    """Assemble an immutable V1 reset batch without action-history overwrite."""

    if not windows:
        raise ValueError("At least one source window is required.")
    horizon = len(windows[0].transition_time_indices)
    if any(len(window.transition_time_indices) != horizon for window in windows):
        raise ValueError("All source windows must have the same horizon.")
    fields = view.fields
    point_snapshots = {
        target: _gather_points(fields[source].float(), windows)
        for target, (source, _width) in V1_SNAPSHOT_FIELDS.items()
    }
    snapshot: dict[str, torch.Tensor | str] = {"snapshot_version": SNAPSHOT_VERSION}
    for target, values in point_snapshots.items():
        # clone() is intentional: no caller can later overwrite V1 action history.
        snapshot[target] = values[:, 0].clone()

    states = _gather_points(fields["states"].float(), windows)
    next_states = _gather_transitions(fields["next_states"].float(), windows)
    expected_rwm = torch.cat([states[:, :1], next_states], dim=1)
    window_continuity_linf = float(
        torch.max(torch.abs(next_states - states[:, 1:]))
    )
    expected_physical = torch.cat(
        [
            point_snapshots["root_state_local"],
            point_snapshots["joint_position"],
            point_snapshots["joint_velocity"],
        ],
        dim=-1,
    )
    transition_commands = _gather_transitions(fields["commands"].float(), windows)
    snapshot_transition_commands = point_snapshots["command"][:, :-1]
    command_source_linf = float(
        torch.max(torch.abs(transition_commands - snapshot_transition_commands))
    )
    assembly = {
        "snapshot_action_source": "sim_action_histories",
        "snapshot_action_overwritten_from_prev_actions": False,
        "transition_command_vs_snapshot_command_linf": command_source_linf,
        "transition_command_source_tolerance": float(command_source_tolerance),
        "transition_command_source_passed": bool(
            command_source_linf <= command_source_tolerance
        ),
        "rwm_next_state_vs_next_row_state_linf": window_continuity_linf,
        "rwm_next_state_vs_next_row_state_passed": bool(
            window_continuity_linf <= SOURCE_CONTINUITY_TOLERANCE
        ),
        "action_sequence_sha256": _canonical_sha256(
            _gather_transitions(fields["actions"].float(), windows)
        ),
        "command_sequence_sha256": _canonical_sha256(transition_commands),
        "snapshot_history_sha256": _canonical_sha256(
            {
                "action": point_snapshots["action"],
                "prev_action": point_snapshots["prev_action"],
                "prev_prev_action": point_snapshots["prev_prev_action"],
            }
        ),
    }
    request = ResetValidationRequest(
        snapshot=snapshot,
        actions=_gather_transitions(fields["actions"].float(), windows),
        transition_commands=transition_commands,
        expected_physical_states=expected_physical,
        expected_rwm_states=expected_rwm,
        expected_action_histories=point_snapshots["action"],
        expected_prev_action_histories=point_snapshots["prev_action"],
        expected_prev_prev_action_histories=point_snapshots["prev_prev_action"],
        expected_commands=point_snapshots["command"],
        expected_contacts=_gather_transitions(fields["contacts"].bool(), windows),
        expected_terminations=_gather_transitions(
            fields["terminations"].bool(), windows
        ).squeeze(-1),
        expected_rewards=(
            _gather_transitions(fields["rewards"].float(), windows).squeeze(-1)
            if "rewards" in fields
            else None
        ),
        windows=tuple(windows),
        horizon=horizon,
    )
    return request, assembly


def _coerce_runner_output(
    value: ResetRunnerOutput | Mapping[str, Any],
) -> ResetRunnerOutput:
    if isinstance(value, ResetRunnerOutput):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Reset runner must return ResetRunnerOutput or a mapping.")
    required = (
        "physical_states",
        "rwm_states",
        "action_histories",
        "prev_action_histories",
        "prev_prev_action_histories",
        "commands",
        "contacts",
        "terminations",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Reset runner output is missing: {missing}")
    return ResetRunnerOutput(
        physical_states=torch.as_tensor(value["physical_states"]),
        rwm_states=torch.as_tensor(value["rwm_states"]),
        action_histories=torch.as_tensor(value["action_histories"]),
        prev_action_histories=torch.as_tensor(value["prev_action_histories"]),
        prev_prev_action_histories=torch.as_tensor(
            value["prev_prev_action_histories"]
        ),
        commands=torch.as_tensor(value["commands"]),
        contacts=torch.as_tensor(value["contacts"]),
        terminations=torch.as_tensor(value["terminations"]),
        rewards=(
            None
            if value.get("rewards") is None
            else torch.as_tensor(value["rewards"])
        ),
        metadata=value.get("metadata"),
    )


def _validate_runner_shapes(
    request: ResetValidationRequest,
    actual: ResetRunnerOutput,
) -> None:
    expected_shapes = {
        "physical_states": tuple(request.expected_physical_states.shape),
        "rwm_states": tuple(request.expected_rwm_states.shape),
        "action_histories": tuple(request.expected_action_histories.shape),
        "prev_action_histories": tuple(
            request.expected_prev_action_histories.shape
        ),
        "prev_prev_action_histories": tuple(
            request.expected_prev_prev_action_histories.shape
        ),
        "commands": tuple(request.expected_commands.shape),
        "contacts": tuple(request.expected_contacts.shape),
        "terminations": tuple(request.expected_terminations.shape),
    }
    errors = {}
    for name, expected_shape in expected_shapes.items():
        got_shape = tuple(torch.as_tensor(getattr(actual, name)).shape)
        if got_shape != expected_shape:
            errors[name] = {"expected": expected_shape, "actual": got_shape}
    if actual.rewards is not None:
        expected_reward_shape = (
            tuple(request.expected_rewards.shape)
            if request.expected_rewards is not None
            else (len(request.windows), request.horizon)
        )
        if tuple(actual.rewards.shape) != expected_reward_shape:
            errors["rewards"] = {
                "expected": expected_reward_shape,
                "actual": tuple(actual.rewards.shape),
            }
    if errors:
        raise ValueError(f"Reset runner output shape mismatch: {errors}")


def _continuous_error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    blocks: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    absolute = torch.abs(actual - expected)
    flat = absolute.reshape(-1)
    result: dict[str, Any] = {
        "all_finite": finite,
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
        "global_linf": float(flat.max()),
        "per_step_linf": [
            float(value)
            for value in absolute.reshape(absolute.shape[0], absolute.shape[1], -1)
            .amax(dim=(0, 2))
            .tolist()
        ],
        "per_window_linf": [
            float(value)
            for value in absolute.reshape(absolute.shape[0], -1).amax(dim=1).tolist()
        ],
    }
    if blocks:
        result["blocks"] = {
            name: {
                "mean": float(absolute[..., start:stop].mean()),
                "median": float(absolute[..., start:stop].median()),
                "p95": float(
                    torch.quantile(
                        absolute[..., start:stop].reshape(-1),
                        0.95,
                    )
                ),
                "max": float(absolute[..., start:stop].max()),
            }
            for name, (start, stop) in blocks.items()
        }
        result["per_step_blocks"] = {
            name: [
                {
                    "mean": float(step_values.mean()),
                    "median": float(step_values.median()),
                    "p95": float(torch.quantile(step_values.reshape(-1), 0.95)),
                    "max": float(step_values.max()),
                }
                for step_values in absolute[..., start:stop].transpose(0, 1)
            ]
            for name, (start, stop) in blocks.items()
        }
    result["per_step"] = [
        {
            "mean": float(step_values.mean()),
            "median": float(step_values.median()),
            "p95": float(torch.quantile(step_values.reshape(-1), 0.95)),
            "max": float(step_values.max()),
        }
        for step_values in absolute.transpose(0, 1)
    ]
    return result


def _exact_error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> dict[str, Any]:
    _require_binary_tensor(actual, name=f"actual {name}")
    _require_binary_tensor(expected, name=f"expected {name}")
    actual = actual.detach().cpu().bool()
    expected = expected.detach().cpu().bool()
    mismatch = actual != expected
    return {
        "exact_match": not bool(mismatch.any()),
        "mismatch_count": int(mismatch.sum()),
        "mismatch_fraction": float(mismatch.float().mean()),
        "per_step_mismatch_count": [
            int(value)
            for value in mismatch.reshape(mismatch.shape[0], mismatch.shape[1], -1)
            .sum(dim=(0, 2))
            .tolist()
        ],
    }


def _worst_continuous(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance: float,
    request: ResetValidationRequest,
) -> dict[str, Any]:
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    error = torch.abs(actual_cpu - expected_cpu)
    flat_index = int(torch.argmax(error))
    unraveled = list(torch.unravel_index(torch.tensor(flat_index), error.shape))
    window_index = int(unraveled[0])
    step = int(unraveled[1])
    trailing = [int(value) for value in unraveled[2:]]
    index = tuple(int(value) for value in unraveled)
    error_value = float(error[index])
    dimension = trailing[0] if len(trailing) == 1 else trailing
    block_name = None
    block_map = (
        PHYSICAL37_BLOCKS
        if name == "physical37"
        else RWM45_BLOCKS if name == "rwm45" else None
    )
    if block_map is not None and isinstance(dimension, int):
        block_name = next(
            (
                candidate
                for candidate, (start, stop) in block_map.items()
                if start <= dimension < stop
            ),
            None,
        )
    transition_step = None if step == 0 else step - 1
    return {
        "signal": name,
        "window_index": window_index,
        "point_step": step,
        "transition_step": transition_step,
        "dimension": dimension,
        "block": block_name,
        "source_window": request.windows[window_index].to_dict(),
        "expected": float(expected_cpu[index]),
        "actual": float(actual_cpu[index]),
        "abs_error": error_value,
        "tolerance": float(tolerance),
        "threshold_ratio": error_value / max(float(tolerance), 1.0e-30),
        "causal_action": (
            None
            if transition_step is None
            else request.actions[window_index, transition_step].tolist()
        ),
        "causal_command": (
            request.expected_commands[window_index, step].tolist()
            if step <= request.horizon
            else None
        ),
        "t0_histories": {
            "action": request.expected_action_histories[window_index, 0].tolist(),
            "prev_action": request.expected_prev_action_histories[
                window_index, 0
            ].tolist(),
            "prev_prev_action": request.expected_prev_prev_action_histories[
                window_index, 0
            ].tolist(),
        },
    }


def _first_divergence(
    request: ResetValidationRequest,
    actual: ResetRunnerOutput,
    *,
    state_tolerance: float,
    history_tolerance: float,
) -> dict[str, Any] | None:
    point_signals = (
        (
            "physical37",
            actual.physical_states,
            request.expected_physical_states,
            state_tolerance,
        ),
        ("rwm45", actual.rwm_states, request.expected_rwm_states, state_tolerance),
        (
            "action_history",
            actual.action_histories,
            request.expected_action_histories,
            history_tolerance,
        ),
        (
            "prev_action_history",
            actual.prev_action_histories,
            request.expected_prev_action_histories,
            history_tolerance,
        ),
        (
            "prev_prev_action_history",
            actual.prev_prev_action_histories,
            request.expected_prev_prev_action_histories,
            history_tolerance,
        ),
        (
            "command",
            actual.commands,
            request.expected_commands,
            history_tolerance,
        ),
    )
    for step in range(request.horizon + 1):
        for name, observed, expected, tolerance in point_signals:
            if (
                step > 0
                and name
                not in {"physical37", "rwm45"}
            ):
                continue
            error = torch.abs(observed[:, step].float().cpu() - expected[:, step].float().cpu())
            bad = error > tolerance
            if bool(bad.any()):
                position = bad.nonzero(as_tuple=False)[0]
                window_index = int(position[0])
                trailing = [int(value) for value in position[1:]]
                index = (window_index, step, *trailing)
                return {
                    "signal": name,
                    "window_index": window_index,
                    "point_step": step,
                    "dimension": trailing[0] if len(trailing) == 1 else trailing,
                    "source_window": request.windows[window_index].to_dict(),
                    "expected": float(expected.detach().float().cpu()[index]),
                    "actual": float(observed.detach().float().cpu()[index]),
                    "abs_error": float(error[tuple(position)]),
                    "tolerance": float(tolerance),
                }
        if step == 0:
            continue
        transition_step = step - 1
        exact_signals = (
            (
                "contact",
                actual.contacts[:, transition_step].bool().cpu(),
                request.expected_contacts[:, transition_step].bool().cpu(),
            ),
            (
                "termination",
                actual.terminations[:, transition_step].bool().cpu(),
                request.expected_terminations[:, transition_step].bool().cpu(),
            ),
        )
        for name, observed, expected in exact_signals:
            bad = observed != expected
            if bool(bad.any()):
                position = bad.nonzero(as_tuple=False)[0]
                window_index = int(position[0])
                trailing = [int(value) for value in position[1:]]
                index = (window_index, *trailing)
                return {
                    "signal": name,
                    "window_index": window_index,
                    "point_step": step,
                    "transition_step": transition_step,
                    "dimension": trailing[0] if len(trailing) == 1 else trailing,
                    "source_window": request.windows[window_index].to_dict(),
                    "expected": bool(expected[index]),
                    "actual": bool(observed[index]),
                    "exact_match_required": True,
                }
    return None


def _align_physical_quaternion_hemisphere(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Align only q/-q equivalence in physical37; no other state alignment."""

    aligned = actual.detach().clone()
    actual_quaternion = aligned[..., 3:7]
    expected_quaternion = expected.detach().to(
        device=aligned.device,
        dtype=aligned.dtype,
    )[..., 3:7]
    flip = torch.sum(actual_quaternion * expected_quaternion, dim=-1) < 0
    aligned[..., 3:7] = torch.where(
        flip.unsqueeze(-1),
        -actual_quaternion,
        actual_quaternion,
    )
    return aligned, int(flip.sum().detach().cpu())


def aggregate_reset_errors(
    request: ResetValidationRequest,
    runner_output: ResetRunnerOutput | Mapping[str, Any],
    *,
    state_tolerance: float = DEFAULT_STATE_TOLERANCE,
    history_command_tolerance: float = DEFAULT_HISTORY_COMMAND_TOLERANCE,
) -> dict[str, Any]:
    """Aggregate all reset/rollout checks; reward never contributes to PASS."""

    validate_gate_tolerances(state_tolerance, history_command_tolerance)
    actual = _coerce_runner_output(runner_output)
    _validate_runner_shapes(request, actual)
    aligned_physical, quaternion_sign_flip_count = (
        _align_physical_quaternion_hemisphere(
            actual.physical_states,
            request.expected_physical_states,
        )
    )
    actual.physical_states = aligned_physical

    continuous = {
        "physical37": (
            actual.physical_states,
            request.expected_physical_states,
            state_tolerance,
            PHYSICAL37_BLOCKS,
        ),
        "rwm45": (
            actual.rwm_states,
            request.expected_rwm_states,
            state_tolerance,
            RWM45_BLOCKS,
        ),
        "action_history": (
            actual.action_histories,
            request.expected_action_histories,
            history_command_tolerance,
            None,
        ),
        "prev_action_history": (
            actual.prev_action_histories,
            request.expected_prev_action_histories,
            history_command_tolerance,
            None,
        ),
        "prev_prev_action_history": (
            actual.prev_prev_action_histories,
            request.expected_prev_prev_action_histories,
            history_command_tolerance,
            None,
        ),
        "command": (
            actual.commands,
            request.expected_commands,
            history_command_tolerance,
            None,
        ),
    }
    metrics: dict[str, Any] = {}
    gate: dict[str, Any] = {
        "state_tolerance": float(state_tolerance),
        "history_command_tolerance": float(history_command_tolerance),
        "reward_is_diagnostic_only": True,
    }
    worst_rows = []
    for name, (observed, expected, tolerance, blocks) in continuous.items():
        summary = _continuous_error_summary(observed, expected, blocks=blocks)
        if name in {
            "action_history",
            "prev_action_history",
            "prev_prev_action_history",
            "command",
        }:
            t0_error = torch.abs(
                observed[:, 0].detach().float().cpu()
                - expected[:, 0].detach().float().cpu()
            )
            summary["gate_scope"] = "t0_only"
            summary["t0_linf"] = float(t0_error.max())
            gate_value = summary["t0_linf"]
        else:
            summary["gate_scope"] = "all_points"
            gate_value = summary["global_linf"]
        metrics[name] = summary
        passed = bool(
            summary["all_finite"] and gate_value <= tolerance
        )
        gate[f"{name}_passed"] = passed
        worst_observed = (
            observed[:, :1]
            if summary["gate_scope"] == "t0_only"
            else observed
        )
        worst_expected = (
            expected[:, :1]
            if summary["gate_scope"] == "t0_only"
            else expected
        )
        worst_rows.append(
            _worst_continuous(
                name,
                worst_observed,
                worst_expected,
                tolerance,
                request,
            )
        )

    metrics["contact"] = _exact_error_summary(
        actual.contacts,
        request.expected_contacts,
        name="contacts",
    )
    metrics["termination"] = _exact_error_summary(
        actual.terminations,
        request.expected_terminations,
        name="terminations",
    )
    discrete_finite = bool(
        torch.isfinite(actual.contacts.detach().float()).all()
        and torch.isfinite(actual.terminations.detach().float()).all()
    )
    metrics["discrete_all_finite"] = discrete_finite
    gate["contact_exact_passed"] = bool(metrics["contact"]["exact_match"])
    gate["termination_exact_passed"] = bool(
        metrics["termination"]["exact_match"]
    )
    gate["discrete_values_finite_passed"] = discrete_finite
    gate["all_continuous_values_finite_passed"] = bool(
        all(metrics[name]["all_finite"] for name in continuous)
    )
    metrics["physical37"]["quaternion_sign_flip_count"] = (
        quaternion_sign_flip_count
    )

    if request.expected_rewards is None or actual.rewards is None:
        metrics["reward"] = {
            "available": False,
            "gated": False,
            "reason": (
                "dataset reward unavailable"
                if request.expected_rewards is None
                else "runner reward unavailable"
            ),
        }
    else:
        reward_summary = _continuous_error_summary(
            actual.rewards.unsqueeze(-1),
            request.expected_rewards.unsqueeze(-1),
        )
        metrics["reward"] = {
            "available": True,
            "gated": False,
            **reward_summary,
        }

    exact_failures = []
    for name, observed, expected in (
        ("contact", actual.contacts, request.expected_contacts),
        ("termination", actual.terminations, request.expected_terminations),
    ):
        mismatch = observed.bool().cpu() != expected.bool().cpu()
        if bool(mismatch.any()):
            position = mismatch.nonzero(as_tuple=False)[0]
            window_index = int(position[0])
            step = int(position[1])
            trailing = [int(value) for value in position[2:]]
            index = tuple(int(value) for value in position)
            exact_failures.append(
                {
                    "signal": name,
                    "window_index": window_index,
                    "transition_step": step,
                    "point_step": step + 1,
                    "dimension": (
                        trailing[0] if len(trailing) == 1 else trailing
                    ),
                    "source_window": request.windows[window_index].to_dict(),
                    "expected": bool(expected.bool().cpu()[index]),
                    "actual": bool(observed.bool().cpu()[index]),
                    "exact_match_required": True,
                }
            )

    failed_gate_values = [
        value
        for key, value in gate.items()
        if key.endswith("_passed")
    ]
    gate["passed"] = bool(all(failed_gate_values))
    first = _first_divergence(
        request,
        actual,
        state_tolerance=state_tolerance,
        history_tolerance=history_command_tolerance,
    )
    worst_continuous = max(
        worst_rows,
        key=lambda row: float(row["threshold_ratio"]),
    )
    worst = exact_failures[0] if exact_failures else worst_continuous
    return {
        "metrics": metrics,
        "gate": gate,
        "first_divergence": first,
        "worst_divergence": worst if not gate["passed"] else None,
        "worst_by_continuous_signal": {
            row["signal"]: row for row in worst_rows
        },
        "runner_metadata": _jsonable(actual.metadata or {}),
    }


def _base_report(
    *,
    dataset_path: str | Path | None,
    dataset_sha256: str | None,
    config: Mapping[str, Any],
    runner_identity: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = implementation_provenance()
    binding = {
        "dataset_sha256": dataset_sha256,
        "config": _jsonable(config),
        "runner_identity": _jsonable(runner_identity),
        "implementation": implementation,
    }
    return {
        "format_version": FORMAT_VERSION,
        "status": "BLOCKED",
        "passed": False,
        "dataset": {
            "path": str(Path(dataset_path).resolve()) if dataset_path else None,
            "sha256": dataset_sha256,
        },
        "config": _jsonable(config),
        "config_sha256": _canonical_sha256(config),
        "runner_identity": _jsonable(runner_identity),
        "implementation": implementation,
        "certification_binding_sha256": _canonical_sha256(binding),
    }


def build_nonpass_report(
    *,
    status: str,
    reason: str,
    dataset_path: str | Path | None,
    dataset_sha256: str | None,
    config: Mapping[str, Any],
    runner_identity: Mapping[str, Any],
    preflight: Mapping[str, Any] | None = None,
    window_selection: Mapping[str, Any] | None = None,
    dataset_runner_binding: Mapping[str, Any] | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    if status not in {"FAIL", "BLOCKED"}:
        raise ValueError("Non-pass report status must be FAIL or BLOCKED.")
    report = _base_report(
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        config=config,
        runner_identity=runner_identity,
    )
    report.update(
        {
            "status": status,
            "passed": False,
            "reason": reason,
            "error_type": error_type,
            "preflight": _jsonable(preflight or {}),
            "window_selection": _jsonable(window_selection or {}),
            "dataset_runner_binding": _jsonable(
                dataset_runner_binding or {}
            ),
            "gate": {"passed": False},
        }
    )
    return report


def validate_gate_tolerances(
    state_tolerance: float,
    history_command_tolerance: float,
) -> None:
    if (
        not math.isfinite(state_tolerance)
        or state_tolerance <= 0.0
        or state_tolerance > MAX_STATE_TOLERANCE
    ):
        raise ValueError(
            f"state_tolerance must be in (0, {MAX_STATE_TOLERANCE}]"
        )
    if (
        not math.isfinite(history_command_tolerance)
        or history_command_tolerance <= 0.0
        or history_command_tolerance > MAX_HISTORY_COMMAND_TOLERANCE
    ):
        raise ValueError(
            "history_command_tolerance must be in "
            f"(0, {MAX_HISTORY_COMMAND_TOLERANCE}]"
        )


def validate_runner_object_identity(
    runner: ResetRunner,
    *,
    runner_identity: Mapping[str, Any],
    allow_test_runner: bool,
) -> None:
    """Bind formal execution to the callable imported from ``FORMAL_RUNNER_SPEC``."""

    if allow_test_runner:
        if runner_identity.get("test_only") is not True:
            raise ValueError(
                "Injected runners require allow_test_runner=true and "
                "runner_identity.test_only=true."
            )
        return
    if runner_identity.get("runner_spec") != FORMAL_RUNNER_SPEC:
        raise ValueError(
            f"formal certification requires runner {FORMAL_RUNNER_SPEC!r}"
        )
    try:
        formal_runner = _load_runner(FORMAL_RUNNER_SPEC)
    except (ImportError, OSError) as exc:
        raise CertificationBlockedError(
            f"formal runner could not be imported: {exc}"
        ) from exc
    if runner is not formal_runner:
        raise ValueError(
            "Formal programmatic certification rejects injected callbacks; "
            "the runner object must be the callable imported from FORMAL_RUNNER_SPEC."
        )


def validate_runner_contract(
    metadata: Mapping[str, Any] | None,
    *,
    runner_identity: Mapping[str, Any],
    runner_config: Mapping[str, Any],
    allow_test_runner: bool,
) -> dict[str, Any]:
    """Reject identity callbacks and incomplete simulator provenance."""

    if allow_test_runner:
        if runner_identity.get("test_only") is not True:
            raise ValueError("allow_test_runner requires runner_identity.test_only=true")
        return {
            "passed": True,
            "test_only": True,
            "formal_v13_scope": False,
            "runtime_scope": "test_only",
        }
    if runner_identity.get("runner_spec") != FORMAL_RUNNER_SPEC:
        raise ValueError(
            f"formal certification requires runner {FORMAL_RUNNER_SPEC!r}"
        )
    if not isinstance(metadata, Mapping):
        raise ValueError("formal runner returned no metadata")
    required_values = {
        "runner_contract_version": "go2_perfect_sim_runner_v1",
        "manual_snapshot_restore": True,
        "automatic_reset_during_rollout": False,
        "snapshot_action_history_overwritten": False,
        "provenance_verified": True,
    }
    for key, expected in required_values.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"formal runner metadata {key!r} must be {expected!r}, "
                f"got {metadata.get(key)!r}"
            )
    for key in ("condition_id", "task"):
        if metadata.get(key) != runner_config.get(key):
            raise ValueError(
                f"runner metadata/config mismatch for {key}: "
                f"{metadata.get(key)!r} != {runner_config.get(key)!r}"
            )
    runtime_scope = runner_config.get("runtime_scope")
    if metadata.get("runtime_scope") != runtime_scope:
        raise ValueError("runner metadata/config runtime_scope mismatch")
    if runtime_scope not in {"v13_canonical", "pinned_v10_adapter"}:
        raise ValueError(f"unsupported runtime_scope: {runtime_scope!r}")
    determinism = metadata.get("determinism")
    if not isinstance(determinism, Mapping) or determinism.get("passed") is not True:
        raise ValueError("formal runner did not pass mandatory double replay")
    if int(determinism.get("repetitions", 0)) < 2:
        raise ValueError("formal runner determinism requires at least two replays")
    if not isinstance(metadata.get("physics"), Mapping):
        raise ValueError("formal runner metadata lacks realized physics")
    physics = metadata["physics"]
    if physics.get("observation_history_at_most_one") is not True:
        raise ValueError(
            "formal runner did not prove observation history_length <= 1"
        )
    if physics.get("observation_delay_disabled") is not True:
        raise ValueError("formal runner did not prove observation delay=0")

    software_versions = metadata.get("software_versions")
    required_software_fields = (
        "python",
        "python_implementation",
        "torch",
        "torch_git_version",
        "cuda_build",
        "mjlab",
        "mujoco",
        "mujoco_warp",
        "warp",
    )
    if (
        not isinstance(software_versions, Mapping)
        or software_versions.get("schema_version")
        != "go2_runner_software_versions_v1"
        or any(
            not str(software_versions.get(key, "")).strip()
            for key in required_software_fields
        )
    ):
        raise ValueError(
            "formal runner software version provenance is incomplete"
        )

    precision = metadata.get("precision")
    if (
        not isinstance(precision, Mapping)
        or precision.get("schema_version") != "go2_runner_precision_v1"
        or precision.get("required_state_action_dtype") != "torch.float32"
        or not isinstance(precision.get("state"), Mapping)
        or not isinstance(precision.get("action"), Mapping)
        or not isinstance(precision.get("model"), Mapping)
    ):
        raise ValueError("formal runner precision provenance is incomplete")

    independent_resets = metadata.get("independent_native_resets")
    if (
        not isinstance(independent_resets, Mapping)
        or independent_resets.get("schema_version")
        != "go2_runner_independent_resets_v1"
        or independent_resets.get("replay_count") != 2
        or independent_resets.get("same_seed") is not True
        or independent_resets.get("seed") != runner_config.get("seed")
        or independent_resets.get("public_native_reset_calls_total") != 2
        or independent_resets.get("clean_start_before_every_replay") is not True
    ):
        raise ValueError(
            "formal runner lacks two independent same-seed public resets"
        )
    if metadata.get("history_burn_in_steps") != runner_config.get(
        "history_burn_in_steps"
    ):
        raise ValueError(
            "formal runner metadata/config history_burn_in_steps mismatch"
        )
    if (
        not isinstance(metadata.get("history_burn_in_steps"), int)
        or isinstance(metadata.get("history_burn_in_steps"), bool)
        or metadata["history_burn_in_steps"] < 3
    ):
        raise ValueError("formal runner requires history_burn_in_steps >= 3")

    runtime_scope_evidence = metadata.get("runtime_scope_evidence")
    if (
        not isinstance(runtime_scope_evidence, Mapping)
        or runtime_scope_evidence.get("runtime_scope") != runtime_scope
        or not str(runtime_scope_evidence.get("repo_commit", "")).strip()
        or not str(runtime_scope_evidence.get("mjlab_commit", "")).strip()
    ):
        raise ValueError("formal runner runtime scope evidence is incomplete")
    device = metadata.get("device_identity")
    if not isinstance(device, Mapping):
        raise ValueError("formal runner metadata lacks physical device identity")
    physical_index = device.get("physical_index")
    if physical_index in {2, 7}:
        raise ValueError(f"formal runner used forbidden physical GPU {physical_index}")
    if physical_index is None or not device.get("uuid") or not device.get("pci_bus_id"):
        raise ValueError("formal runner physical device identity is incomplete")
    return {
        "passed": True,
        "test_only": False,
        "formal_v13_scope": runtime_scope == "v13_canonical",
        "runtime_scope": runtime_scope,
        "runner_contract_version": metadata["runner_contract_version"],
        "determinism": _jsonable(determinism),
        "independent_native_resets": _jsonable(independent_resets),
        "software_versions": _jsonable(software_versions),
        "precision": _jsonable(precision),
        "runtime_scope_evidence": _jsonable(runtime_scope_evidence),
        "device_identity": _jsonable(device),
    }


def run_reset_certification(
    *,
    dataset: Mapping[str, Any],
    dataset_path: str | Path | None,
    dataset_sha256: str | None,
    runner: ResetRunner,
    runner_identity: Mapping[str, Any],
    runner_config: Mapping[str, Any],
    horizon: int,
    num_windows: int,
    seed: int,
    min_source_timestep: int = 0,
    state_tolerance: float = DEFAULT_STATE_TOLERANCE,
    history_command_tolerance: float = DEFAULT_HISTORY_COMMAND_TOLERANCE,
    allow_test_runner: bool = False,
) -> dict[str, Any]:
    """Run the complete data-side certification around an injected runner."""

    validate_gate_tolerances(state_tolerance, history_command_tolerance)
    config = {
        "horizon": int(horizon),
        "num_windows": int(num_windows),
        "seed": int(seed),
        "min_source_timestep": int(min_source_timestep),
        "state_tolerance": float(state_tolerance),
        "history_command_tolerance": float(history_command_tolerance),
        "physical_state_definition": "root_state_local13+joint_position12+joint_velocity12",
        "rwm_state_definition": "go2_rwm_state45",
        "all_h_transitions_nonterminal": True,
        "condition_id": runner_config.get("condition_id"),
        "task": runner_config.get("task"),
        "runner_config_sha256": _canonical_sha256(runner_config),
    }
    runner_horizon = runner_config.get("horizon")
    runner_runtime_scope = runner_config.get("runtime_scope")
    runner_binding_issues = []
    if (
        isinstance(runner_horizon, bool)
        or not isinstance(runner_horizon, int)
        or runner_horizon != int(horizon)
    ):
        runner_binding_issues.append(
            f"runner_config.horizon {runner_horizon!r} does not match "
            f"certification horizon {int(horizon)!r}"
        )
    if not isinstance(runner_runtime_scope, str) or not runner_runtime_scope:
        runner_binding_issues.append("runner_config.runtime_scope is required")
    elif allow_test_runner and runner_runtime_scope != "test_only":
        runner_binding_issues.append(
            "test-only certification requires runner_config.runtime_scope='test_only'"
        )
    elif not allow_test_runner and runner_runtime_scope not in {
        "v13_canonical",
        "pinned_v10_adapter",
    }:
        runner_binding_issues.append(
            f"unsupported formal runner runtime_scope {runner_runtime_scope!r}"
        )
    dataset_runner_binding = _dataset_runner_metadata_binding(
        dataset,
        runner_config,
        required=not allow_test_runner,
    )
    if not dataset_runner_binding["passed"]:
        runner_binding_issues.extend(dataset_runner_binding["issues"])
    if runner_binding_issues:
        return build_nonpass_report(
            status="FAIL",
            reason="; ".join(runner_binding_issues),
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            dataset_runner_binding=dataset_runner_binding,
        )
    view, preflight = preflight_dataset(
        dataset,
        expected_condition_id=(
            str(runner_config["condition_id"])
            if runner_config.get("condition_id") is not None
            else None
        ),
        expected_task=(
            str(runner_config["task"])
            if runner_config.get("task") is not None
            else None
        ),
        expected_horizon=int(horizon),
        expected_runtime_scope=runner_runtime_scope,
    )
    if view is None:
        return build_nonpass_report(
            status="FAIL",
            reason="dataset preflight failed",
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
        )
    try:
        windows, window_report = select_source_windows(
            view,
            horizon=horizon,
            count=num_windows,
            seed=seed,
            min_source_timestep=min_source_timestep,
        )
    except Exception as exc:
        return build_nonpass_report(
            status="FAIL",
            reason=str(exc),
            error_type=type(exc).__name__,
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
        )
    if not window_report["passed"]:
        return build_nonpass_report(
            status="FAIL",
            reason=str(window_report["error"]),
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
    request, assembly = prepare_validation_request(
        view,
        windows,
        command_source_tolerance=history_command_tolerance,
    )
    if not assembly["transition_command_source_passed"]:
        return build_nonpass_report(
            status="FAIL",
            reason=(
                "dataset commands and V1 snapshot commands disagree beyond "
                f"{history_command_tolerance}"
            ),
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
    if not assembly["rwm_next_state_vs_next_row_state_passed"]:
        return build_nonpass_report(
            status="FAIL",
            reason=(
                "selected source windows violate next_states[t] == states[t+1]: "
                f"L∞={assembly['rwm_next_state_vs_next_row_state_linf']}"
            ),
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
    try:
        validate_runner_object_identity(
            runner,
            runner_identity=runner_identity,
            allow_test_runner=allow_test_runner,
        )
        output = runner(request, runner_config)
        coerced_output = _coerce_runner_output(output)
        runner_contract = validate_runner_contract(
            coerced_output.metadata,
            runner_identity=runner_identity,
            runner_config=runner_config,
            allow_test_runner=allow_test_runner,
        )
        aggregate = aggregate_reset_errors(
            request,
            coerced_output,
            state_tolerance=state_tolerance,
            history_command_tolerance=history_command_tolerance,
        )
    except Exception as exc:
        failure_status = _runner_exception_status(exc)
        return build_nonpass_report(
            status=failure_status,
            reason=f"reset runner failed: {exc}",
            error_type=type(exc).__name__,
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
    report = _base_report(
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        config=config,
        runner_identity={
            **dict(runner_identity),
            "runtime_metadata": aggregate["runner_metadata"],
        },
    )
    numerical_gate_passed = bool(aggregate["gate"]["passed"])
    formal_v13_scope = bool(runner_contract.get("formal_v13_scope", False))
    passed = bool(
        numerical_gate_passed
        and not allow_test_runner
        and formal_v13_scope
    )
    status = (
        "PASS"
        if passed
        else "TEST_ONLY_PASS"
        if numerical_gate_passed and allow_test_runner
        else "BLOCKED"
        if numerical_gate_passed and not formal_v13_scope
        else "FAIL"
    )
    report.update(
        {
            "status": status,
            "passed": passed,
            "reason": (
                "all reset and H-step rollout gates passed"
                if passed
                else "test-only numerical gate passed; not a formal certificate"
                if status == "TEST_ONLY_PASS"
                else (
                    "numerical reset gate passed in a pinned V10 adapter, but "
                    "canonical V13 runtime provenance is unavailable"
                )
                if status == "BLOCKED"
                else "one or more reset/rollout gates failed"
            ),
            "development_numerical_gate_passed": numerical_gate_passed,
            "preflight": preflight,
            "window_selection": window_report,
            "request_assembly": assembly,
            "dataset_runner_binding": dataset_runner_binding,
            "runner_contract": runner_contract,
            "metrics": aggregate["metrics"],
            "gate": aggregate["gate"],
            "first_divergence": aggregate["first_divergence"],
            "worst_divergence": aggregate["worst_divergence"],
            "worst_by_continuous_signal": aggregate[
                "worst_by_continuous_signal"
            ],
            "runner_metadata": aggregate["runner_metadata"],
        }
    )
    report["certification_binding_sha256"] = _canonical_sha256(
        {
            "dataset_sha256": dataset_sha256,
            "config": config,
            "runner_identity": report["runner_identity"],
            "selected_windows": window_report["selected_windows"],
            "request_assembly": assembly,
            "dataset_runner_binding": dataset_runner_binding,
            "implementation": report["implementation"],
        }
    )
    return report


def _markdown_report(report: Mapping[str, Any]) -> str:
    gate = dict(report.get("gate") or {})
    metrics = dict(report.get("metrics") or {})
    lines = [
        "# Go2 source-reset certification",
        "",
        f"- Status: **{report.get('status', 'UNKNOWN')}**",
        f"- Passed: `{bool(report.get('passed', False))}`",
        f"- Reason: {report.get('reason', '')}",
        f"- Dataset: `{(report.get('dataset') or {}).get('path')}`",
        f"- Dataset SHA256: `{(report.get('dataset') or {}).get('sha256')}`",
        f"- Binding SHA256: `{report.get('certification_binding_sha256')}`",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for key, value in sorted(gate.items()):
        if key.endswith("_passed") or key == "passed":
            lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Continuous global L∞", "", "| Signal | L∞ |", "|---|---:|"])
    for name in (
        "physical37",
        "rwm45",
        "action_history",
        "prev_action_history",
        "prev_prev_action_history",
        "command",
    ):
        if name in metrics:
            lines.append(f"| `{name}` | `{metrics[name].get('global_linf')}` |")
    lines.extend(["", "## First divergence", "", "```json"])
    lines.append(
        json.dumps(report.get("first_divergence"), indent=2, sort_keys=True)
    )
    lines.extend(["```", "", "## Worst divergence", "", "```json"])
    lines.append(
        json.dumps(report.get("worst_divergence"), indent=2, sort_keys=True)
    )
    lines.extend(["```", ""])
    return "\n".join(lines)


def write_certification_artifacts(
    report: Mapping[str, Any],
    output_json: str | Path,
) -> tuple[Path, Path]:
    """Atomically write hash-bound JSON and human-readable Markdown."""

    json_path = Path(output_json).expanduser().resolve()
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        json_path: json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        md_path: _markdown_report(report),
    }
    for path, payload in payloads.items():
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    return json_path, md_path


def _load_runner(spec: str) -> ResetRunner:
    if ":" not in spec:
        raise ValueError("--runner must use module:callable syntax")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    runner = getattr(module, attribute)
    if not callable(runner):
        raise TypeError(f"Runner {spec!r} is not callable.")
    return runner


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition-id",
        required=True,
        choices=("g0", "p5", "p75", "rr05", "rr03"),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--runner",
        default=None,
        help="Perfect-simulator adapter as module:callable. Missing runner writes BLOCKED.",
    )
    parser.add_argument(
        "--runner-config-json",
        default=None,
        help="JSON file passed unchanged to the runner and included in the binding.",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9102)
    parser.add_argument("--min-source-timestep", type=int, default=0)
    parser.add_argument("--state-tolerance", type=float, default=DEFAULT_STATE_TOLERANCE)
    parser.add_argument(
        "--history-command-tolerance",
        type=float,
        default=DEFAULT_HISTORY_COMMAND_TOLERANCE,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    config = {
        "horizon": int(args.horizon),
        "num_windows": int(args.num_windows),
        "seed": int(args.seed),
        "min_source_timestep": int(args.min_source_timestep),
        "state_tolerance": float(args.state_tolerance),
        "history_command_tolerance": float(
            args.history_command_tolerance
        ),
        "condition_id": str(args.condition_id),
        "task": str(args.task),
    }
    runner_config: dict[str, Any] = {}
    runner_identity: dict[str, Any] = {"runner_spec": args.runner}
    dataset_hash = None

    try:
        validate_gate_tolerances(
            args.state_tolerance,
            args.history_command_tolerance,
        )
    except ValueError as exc:
        report = build_nonpass_report(
            status="FAIL",
            reason=str(exc),
            error_type=type(exc).__name__,
            dataset_path=dataset_path,
            dataset_sha256=None,
            config=config,
            runner_identity=runner_identity,
        )
        write_certification_artifacts(report, output_path)
        return 3

    if not dataset_path.is_file():
        report = build_nonpass_report(
            status="BLOCKED",
            reason=f"dataset does not exist: {dataset_path}",
            error_type="FileNotFoundError",
            dataset_path=dataset_path,
            dataset_sha256=None,
            config=config,
            runner_identity=runner_identity,
        )
        write_certification_artifacts(report, output_path)
        return 2
    try:
        dataset_hash = sha256_path(dataset_path)
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        report = build_nonpass_report(
            status="BLOCKED",
            reason=f"dataset could not be loaded: {exc}",
            error_type=type(exc).__name__,
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
        )
        write_certification_artifacts(report, output_path)
        return 2
    if not isinstance(dataset, Mapping):
        report = build_nonpass_report(
            status="FAIL",
            reason="loaded dataset is not a mapping",
            error_type="TypeError",
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
        )
        write_certification_artifacts(report, output_path)
        return 3

    if args.runner_config_json:
        runner_config_path = Path(args.runner_config_json).expanduser().resolve()
        if not runner_config_path.is_file():
            report = build_nonpass_report(
                status="BLOCKED",
                reason=f"runner config does not exist: {runner_config_path}",
                error_type="FileNotFoundError",
                dataset_path=dataset_path,
                dataset_sha256=dataset_hash,
                config=config,
                runner_identity=runner_identity,
            )
            write_certification_artifacts(report, output_path)
            return 2
        try:
            runner_config = json.loads(
                runner_config_path.read_text(encoding="utf-8")
            )
            if not isinstance(runner_config, dict):
                raise TypeError("runner config must be a JSON object")
            runner_identity["runner_config_path"] = str(runner_config_path)
            runner_identity["runner_config_sha256"] = sha256_path(
                runner_config_path
            )
            if runner_config.get("condition_id") != args.condition_id:
                raise ValueError(
                    "runner config condition_id does not match --condition-id"
                )
            if runner_config.get("task") != args.task:
                raise ValueError("runner config task does not match --task")
        except Exception as exc:
            report = build_nonpass_report(
                status="BLOCKED",
                reason=f"runner config could not be loaded: {exc}",
                error_type=type(exc).__name__,
                dataset_path=dataset_path,
                dataset_sha256=dataset_hash,
                config=config,
                runner_identity=runner_identity,
            )
            write_certification_artifacts(report, output_path)
            return 2

    view, preflight = preflight_dataset(
        dataset,
        expected_condition_id=args.condition_id,
        expected_task=args.task,
        expected_horizon=args.horizon,
        expected_runtime_scope=(
            str(runner_config["runtime_scope"])
            if runner_config.get("runtime_scope") is not None
            else None
        ),
    )
    if view is None:
        report = build_nonpass_report(
            status="FAIL",
            reason="dataset preflight failed",
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
        )
        write_certification_artifacts(report, output_path)
        return 3
    windows, window_report = select_source_windows(
        view,
        horizon=args.horizon,
        count=args.num_windows,
        seed=args.seed,
        min_source_timestep=args.min_source_timestep,
    )
    if not window_report["passed"]:
        report = build_nonpass_report(
            status="FAIL",
            reason=str(window_report["error"]),
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
        write_certification_artifacts(report, output_path)
        return 3
    if not args.runner:
        report = build_nonpass_report(
            status="BLOCKED",
            reason="perfect-simulator runner was not provided",
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
        write_certification_artifacts(report, output_path)
        return 2
    if args.runner != FORMAL_RUNNER_SPEC:
        report = build_nonpass_report(
            status="BLOCKED",
            reason=(
                f"formal certification requires runner {FORMAL_RUNNER_SPEC!r}; "
                f"got {args.runner!r}"
            ),
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
        write_certification_artifacts(report, output_path)
        return 2
    try:
        runner = _load_runner(args.runner)
    except Exception as exc:
        report = build_nonpass_report(
            status="BLOCKED",
            reason=f"perfect-simulator runner could not be loaded: {exc}",
            error_type=type(exc).__name__,
            dataset_path=dataset_path,
            dataset_sha256=dataset_hash,
            config=config,
            runner_identity=runner_identity,
            preflight=preflight,
            window_selection=window_report,
        )
        write_certification_artifacts(report, output_path)
        return 2

    report = run_reset_certification(
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_sha256=dataset_hash,
        runner=runner,
        runner_identity=runner_identity,
        runner_config=runner_config,
        horizon=args.horizon,
        num_windows=args.num_windows,
        seed=args.seed,
        min_source_timestep=args.min_source_timestep,
        state_tolerance=args.state_tolerance,
        history_command_tolerance=args.history_command_tolerance,
    )
    write_certification_artifacts(report, output_path)
    if report["status"] == "PASS":
        return 0
    return 2 if report["status"] == "BLOCKED" else 3


if __name__ == "__main__":
    sys.exit(main())
