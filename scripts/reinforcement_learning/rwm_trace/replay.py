"""Mutable TRACE replay and within-synthetic mixing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .schemas import REPLAY_SCHEMA_HASH, REPLAY_SCHEMA_VERSION


REPLAY_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)


def _validate_batch(batch: Mapping[str, torch.Tensor], *, allow_empty: bool = False) -> int:
    size: int | None = None
    for key in REPLAY_KEYS:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Replay batch is missing tensor {key!r}.")
        if value.ndim < 1:
            raise ValueError(f"Replay tensor {key!r} has no batch dimension.")
        if size is None:
            size = int(value.shape[0])
        elif int(value.shape[0]) != size:
            raise ValueError("Replay tensors have inconsistent leading dimensions.")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Replay tensor {key!r} contains NaN/Inf.")
    result = int(size or 0)
    if result == 0 and not allow_empty:
        raise ValueError("Replay batch is empty.")
    return result


class MutableTraceReplayBuffer:
    """CPU FIFO/ring replay with reproducible sampler state.

    Rule/learned selection scores and trajectory identities are stored as audit
    metadata and never returned in the agent training batch.
    """

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if int(capacity) < 1:
            raise ValueError("TRACE replay capacity must be positive.")
        self.capacity = int(capacity)
        self._storage: dict[str, torch.Tensor] = {}
        self._size = 0
        self._write_index = 0
        self._audit_rows: list[dict[str, Any]] = []
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.total_inserted = 0
        self.path = "mutable://go2-online-trace"
        self.sha256 = REPLAY_SCHEMA_HASH
        self.metadata: dict[str, Any] = {
            "format_version": REPLAY_SCHEMA_VERSION,
            "schema_sha256": REPLAY_SCHEMA_HASH,
            "mutable": True,
        }

    def __len__(self) -> int:
        return self._size

    @property
    def _data(self) -> dict[str, torch.Tensor]:
        """Return a chronological compatibility view of the ring contents."""

        if self._size == 0:
            return {}
        if self._size < self.capacity:
            return {
                key: value[: self._size]
                for key, value in self._storage.items()
            }
        if self._write_index == 0:
            return dict(self._storage)
        return {
            key: torch.cat(
                (value[self._write_index :], value[: self._write_index]),
                dim=0,
            )
            for key, value in self._storage.items()
        }

    @property
    def num_transitions(self) -> int:
        return len(self)

    def bind_provenance(self, metadata: Mapping[str, Any]) -> None:
        required = {"gamma", "n_step", "reward_config_sha256"}
        if not required <= set(metadata):
            raise ValueError(
                f"Mutable TRACE provenance is missing {sorted(required-set(metadata))}."
            )
        self.metadata = {**self.metadata, **dict(metadata)}

    @property
    def audit_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._audit_rows)

    def add_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        trajectory_id: str,
        learned_score: float,
        refresh_step: int,
        score_source: str = "learned",
    ) -> int:
        return self.add_batches(
            [batch],
            trajectory_ids=[trajectory_id],
            scores=[learned_score],
            refresh_step=refresh_step,
            score_source=score_source,
        )

    def add_batches(
        self,
        batches: Sequence[Mapping[str, torch.Tensor]],
        *,
        trajectory_ids: Sequence[str],
        scores: Sequence[float],
        refresh_step: int,
        score_source: str = "learned",
    ) -> int:
        """Append a proposal event with one concatenation and ring-buffer write."""

        if not batches:
            return 0
        if len(batches) != len(trajectory_ids) or len(batches) != len(scores):
            raise ValueError("TRACE replay batch metadata lengths are inconsistent.")
        if score_source not in {"learned", "rule_bootstrap"}:
            raise ValueError("TRACE score_source is unsupported.")

        converted: list[dict[str, torch.Tensor]] = []
        counts: list[int] = []
        for batch in batches:
            counts.append(_validate_batch(batch))
            converted.append(
                {
                    key: batch[key].detach().to("cpu").contiguous()
                    for key in REPLAY_KEYS
                }
            )
        reference = converted[0]
        for batch in converted[1:]:
            for key in REPLAY_KEYS:
                if batch[key].shape[1:] != reference[key].shape[1:]:
                    raise ValueError(f"TRACE replay shape changed for {key!r}.")
                if batch[key].dtype != reference[key].dtype:
                    raise ValueError(f"TRACE replay dtype changed for {key!r}.")
        if self._storage:
            for key in REPLAY_KEYS:
                if self._storage[key].shape[1:] != reference[key].shape[1:]:
                    raise ValueError(f"TRACE replay shape changed for {key!r}.")
                if self._storage[key].dtype != reference[key].dtype:
                    raise ValueError(f"TRACE replay dtype changed for {key!r}.")
        else:
            self._storage = {
                key: torch.empty(
                    (self.capacity, *value.shape[1:]),
                    dtype=value.dtype,
                    device="cpu",
                )
                for key, value in reference.items()
            }

        incoming = {
            key: torch.cat([batch[key] for batch in converted], dim=0)
            for key in REPLAY_KEYS
        }
        total_count = sum(counts)
        if total_count >= self.capacity:
            for key in REPLAY_KEYS:
                self._storage[key].copy_(incoming[key][-self.capacity :])
            self._size = self.capacity
            self._write_index = 0
        else:
            first = min(total_count, self.capacity - self._write_index)
            second = total_count - first
            for key in REPLAY_KEYS:
                self._storage[key][
                    self._write_index : self._write_index + first
                ].copy_(incoming[key][:first])
                if second:
                    self._storage[key][:second].copy_(incoming[key][first:])
            self._write_index = (self._write_index + total_count) % self.capacity
            self._size = min(self.capacity, self._size + total_count)

        audit_rows: list[dict[str, Any]] = []
        score_key = (
            "learned_score" if score_source == "learned" else "rule_score"
        )
        for count, trajectory_id, learned_score in zip(
            counts, trajectory_ids, scores, strict=True
        ):
            for index in range(count):
                audit_rows.append(
                    {
                        "trajectory_id": str(trajectory_id),
                        "selection_score": float(learned_score),
                        "score_source": str(score_source),
                        "refresh_step": int(refresh_step),
                        "trajectory_transition_index": index,
                        score_key: float(learned_score),
                    }
                )
        self._audit_rows.extend(audit_rows)
        overflow = max(0, len(self._audit_rows) - self.capacity)
        if overflow:
            del self._audit_rows[:overflow]
        self.total_inserted += total_count
        return total_count

    def sample(
        self,
        count: int,
        *,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        if int(count) < 1:
            raise ValueError("TRACE replay sample count must be positive.")
        if len(self) == 0:
            raise ValueError("TRACE replay is empty.")
        indices = torch.randint(
            len(self), (int(count),), generator=self._generator, device="cpu"
        )
        return {
            key: value[indices].to(device, non_blocking=True)
            for key, value in self._storage.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": REPLAY_SCHEMA_VERSION,
            "schema_sha256": REPLAY_SCHEMA_HASH,
            "capacity": self.capacity,
            "data": self._data,
            "audit_rows": self._audit_rows,
            "generator_state": self._generator.get_state(),
            "total_inserted": self.total_inserted,
            "metadata": self.metadata,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError("TRACE replay checkpoint format mismatch.")
        if state.get("schema_sha256") != REPLAY_SCHEMA_HASH:
            raise ValueError("TRACE replay checkpoint schema mismatch.")
        if int(state["capacity"]) != self.capacity:
            raise ValueError("TRACE replay checkpoint capacity mismatch.")
        data = state["data"]
        if data:
            _validate_batch(data, allow_empty=True)
            size = int(data["reward"].shape[0])
            if size > self.capacity:
                raise ValueError("TRACE replay checkpoint exceeds configured capacity.")
            self._storage = {
                key: torch.empty(
                    (self.capacity, *data[key].shape[1:]),
                    dtype=data[key].dtype,
                    device="cpu",
                )
                for key in REPLAY_KEYS
            }
            for key in REPLAY_KEYS:
                self._storage[key][:size].copy_(
                    data[key].detach().cpu().contiguous()
                )
            self._size = size
            self._write_index = size % self.capacity
        else:
            self._storage = {}
            self._size = 0
            self._write_index = 0
        self._audit_rows = [dict(row) for row in state["audit_rows"]]
        if len(self._audit_rows) != len(self):
            raise ValueError("TRACE replay audit-row count mismatch.")
        self._generator.set_state(state["generator_state"].detach().cpu())
        self.total_inserted = int(state["total_inserted"])
        if dict(state.get("metadata") or {}) != self.metadata:
            raise ValueError("TRACE replay provenance changed across resume.")

    def save(self, path: str | Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)

    @classmethod
    def load(
        cls, path: str | Path, *, expected_capacity: int | None = None
    ) -> "MutableTraceReplayBuffer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        capacity = int(state["capacity"])
        if expected_capacity is not None and capacity != int(expected_capacity):
            raise ValueError("TRACE replay artifact capacity differs from configuration.")
        result = cls(capacity)
        result.load_state_dict(state)
        return result


def _half_up(value: float) -> int:
    return int(torch.floor(torch.tensor(float(value)) + 0.5).item())


def mix_trace_within_synthetic(
    rwm_synthetic_batch: Mapping[str, torch.Tensor],
    trace_buffer: MutableTraceReplayBuffer | None,
    *,
    trace_ratio_within_synthetic: float,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Mix TRACE only into the RWM/simulator synthetic sub-batch.

    Real replay is intentionally absent from this API, preventing accidental
    interpretation of the TRACE ratio as a fraction of the total agent batch.
    """

    ratio = float(trace_ratio_within_synthetic)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("TRACE-within-synthetic ratio must be in [0,1].")
    synthetic_count = _validate_batch(rwm_synthetic_batch)
    trace_count = _half_up(synthetic_count * ratio)
    rwm_count = synthetic_count - trace_count
    if trace_count and (trace_buffer is None or len(trace_buffer) == 0):
        raise ValueError("Configured TRACE samples require a non-empty mutable buffer.")
    rwm = {
        key: value[:rwm_count]
        for key, value in rwm_synthetic_batch.items()
        if key in REPLAY_KEYS
    }
    if trace_count:
        trace = trace_buffer.sample(
            trace_count, device=rwm_synthetic_batch["reward"].device
        )
        mixed = {
            key: torch.cat((rwm[key], trace[key].to(dtype=rwm[key].dtype)), dim=0)
            for key in REPLAY_KEYS
        }
    else:
        mixed = {key: rwm[key].clone() for key in REPLAY_KEYS}
    if shuffle and synthetic_count > 1:
        permutation = torch.randperm(
            synthetic_count,
            generator=generator,
            device="cpu",
        ).to(mixed["reward"].device)
        mixed = {key: value[permutation] for key, value in mixed.items()}
    return mixed, {
        "synthetic_count": synthetic_count,
        "rwm_count": rwm_count,
        "trace_count": trace_count,
    }


class TraceReplaySampler:
    """Legacy read-only artifact sampler; not used by the formal online path."""

    def __init__(self, path: str | Path, *, seed: int = 0) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("format_version") not in {
            "go2_trace_replay_v1",
            "go2_trace_replay_v2",
        }:
            raise ValueError("Unsupported legacy TRACE replay artifact.")
        self.data = {key: artifact[key].detach().cpu() for key in REPLAY_KEYS}
        self.num_transitions = _validate_batch(self.data)
        self.metadata = dict(artifact.get("metadata") or {})
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def sample(
        self, count: int, *, device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        indices = torch.randint(
            self.num_transitions,
            (int(count),),
            generator=self.generator,
        )
        return {key: value[indices].to(device) for key, value in self.data.items()}


def mix_trace_replay_batch(
    batch: Mapping[str, torch.Tensor],
    sampler: TraceReplaySampler | None,
    ratio: float,
) -> tuple[dict[str, torch.Tensor], int]:
    """Legacy static replacement helper retained only for old callers."""

    if sampler is None or ratio <= 0.0:
        return dict(batch), 0
    count = _half_up(int(batch["reward"].shape[0]) * float(ratio))
    if count == 0:
        return dict(batch), 0
    trace = sampler.sample(count, device=batch["reward"].device)
    mixed = dict(batch)
    for key in REPLAY_KEYS:
        mixed[key] = batch[key].clone()
        mixed[key][:count] = trace[key].to(dtype=batch[key].dtype)
    return mixed, count
