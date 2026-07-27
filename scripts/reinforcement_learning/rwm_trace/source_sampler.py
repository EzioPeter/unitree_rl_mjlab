"""Deterministic V1 snapshot sampling from the active V13 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .artifact_manifest import sha256_path
from .simulator_reset import SNAPSHOT_VERSION, validate_snapshot


_DATASET_TO_SNAPSHOT = {
    "root_state_local": "sim_root_states_local",
    "joint_position": "sim_joint_positions",
    "joint_velocity": "sim_joint_velocities",
    "action": "sim_action_histories",
    "prev_action": "sim_prev_action_histories",
    "prev_prev_action": "sim_prev_prev_action_histories",
    "command": "sim_snapshot_commands",
}


def _stack(dataset: Mapping[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        result = value
    elif isinstance(value, list) and value and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        result = torch.stack(value)
    else:
        raise ValueError(f"V13 dataset field {key!r} is absent or malformed.")
    if result.ndim < 2:
        raise ValueError(f"V13 dataset field {key!r} must begin [time,env].")
    return result.detach().cpu()


class V13SnapshotSourceSampler:
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        expected_dataset_sha256: str,
        condition_id: str,
        seed: int,
    ) -> None:
        self.path = Path(dataset_path).expanduser().resolve()
        self.sha256 = sha256_path(self.path)
        if self.sha256 != expected_dataset_sha256:
            raise ValueError("Active V13 dataset hash does not match TRACE config.")
        self.dataset = torch.load(self.path, map_location="cpu", weights_only=False)
        metadata = dict(self.dataset.get("metadata") or {})
        actual_condition = metadata.get("condition_id", metadata.get("condition"))
        if isinstance(actual_condition, Mapping):
            actual_condition = actual_condition.get("condition_id")
        if str(actual_condition) != str(condition_id):
            raise ValueError(
                f"Dataset condition {actual_condition!r} does not match {condition_id!r}."
            )
        snapshot_version = metadata.get(
            "simulator_snapshot_version", metadata.get("snapshot_version")
        )
        if snapshot_version != SNAPSHOT_VERSION:
            raise ValueError("Active V13 dataset does not declare V1 simulator snapshots.")
        self._snapshot = {
            target: _stack(self.dataset, source)
            for target, source in _DATASET_TO_SNAPSHOT.items()
        }
        prefix = next(iter(self._snapshot.values())).shape[:2]
        if any(value.shape[:2] != prefix for value in self._snapshot.values()):
            raise ValueError("V13 snapshot fields have inconsistent [time,env] shapes.")
        valid = (
            _stack(self.dataset, "trace_valid_masks").bool()
            if self.dataset.get("trace_valid_masks") is not None
            else torch.ones(prefix, dtype=torch.bool)
        )
        if valid.shape != prefix:
            raise ValueError("trace_valid_masks shape differs from snapshot fields.")
        self._eligible = valid.nonzero(as_tuple=False)
        if len(self._eligible) == 0:
            raise ValueError("Active V13 dataset has no trace-valid snapshot rows.")
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.sample_count = 0

    def sample(self, count: int) -> tuple[dict[str, Any], list[str]]:
        if int(count) < 1:
            raise ValueError("Snapshot source count must be positive.")
        if int(count) > len(self._eligible):
            raise ValueError("Requested more unique source rows than are eligible.")
        selected = self._eligible[
            torch.randperm(len(self._eligible), generator=self.generator)[: int(count)]
        ]
        snapshot: dict[str, Any] = {"snapshot_version": SNAPSHOT_VERSION}
        for key, values in self._snapshot.items():
            snapshot[key] = torch.stack(
                [values[int(t), int(env)] for t, env in selected.tolist()]
            ).float()
        validate_snapshot(snapshot, int(count))
        ids = [
            f"{self.sha256[:16]}:{int(t)}:{int(env)}"
            for t, env in selected.tolist()
        ]
        self.sample_count += int(count)
        return snapshot, ids

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.path),
            "dataset_sha256": self.sha256,
            "generator_state": self.generator.get_state(),
            "sample_count": self.sample_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["dataset_path"] != str(self.path) or state["dataset_sha256"] != self.sha256:
            raise ValueError("Snapshot source changed across resume.")
        self.generator.set_state(state["generator_state"].detach().cpu())
        self.sample_count = int(state["sample_count"])
