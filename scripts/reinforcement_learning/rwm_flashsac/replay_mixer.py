"""Strict three-source replay mixing for V12 FlashSAC.

This module has no dependency on the trainer.  It samples real, RWM, and TRACE
replay with explicit source counts, validates their shared schema, concatenates
them, and shuffles the result.  Missing sources and shape mismatches fail
closed; there is no fallback or ratio renormalization.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


REPLAY_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)
SUPPORTED_EXTERNAL_FORMATS = {
    "go2_real_replay_v1",
    "go2_trace_replay_v1",
    "go2_trace_replay_v2",
}
SOURCE_REAL = 0
SOURCE_RWM = 1
SOURCE_SIM = 2
SyntheticMode = Literal["rwm", "trace", "mixed"]


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayMixConfig:
    batch_size: int
    real_ratio: float = 0.05
    synthetic_mode: SyntheticMode = "rwm"
    trace_ratio_within_synthetic: float = 0.0
    shuffle: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0.0 <= self.real_ratio < 1.0:
            raise ValueError("real_ratio must be in [0, 1).")
        if self.synthetic_mode not in {"rwm", "trace", "mixed"}:
            raise ValueError(f"Unknown synthetic_mode={self.synthetic_mode!r}.")
        ratio = float(self.trace_ratio_within_synthetic)
        if self.synthetic_mode == "rwm" and ratio != 0.0:
            raise ValueError("synthetic_mode='rwm' requires trace_ratio_within_synthetic=0.")
        if self.synthetic_mode == "trace" and ratio != 1.0:
            raise ValueError("synthetic_mode='trace' requires trace_ratio_within_synthetic=1.")
        if self.synthetic_mode == "mixed" and not 0.0 < ratio < 1.0:
            raise ValueError("synthetic_mode='mixed' requires a trace ratio strictly between 0 and 1.")


@dataclass(frozen=True)
class ReplaySourceCounts:
    real: int
    rwm: int
    sim: int

    @property
    def total(self) -> int:
        return self.real + self.rwm + self.sim


def compute_source_counts(config: ReplayMixConfig) -> ReplaySourceCounts:
    config.validate()
    real = int(config.batch_size * config.real_ratio)
    if config.real_ratio > 0.0 and real == 0:
        raise ValueError("batch_size is too small to include the configured real replay fraction.")
    synthetic = config.batch_size - real
    sim = int(synthetic * config.trace_ratio_within_synthetic)
    rwm = synthetic - sim
    counts = ReplaySourceCounts(real=real, rwm=rwm, sim=sim)
    if counts.total != config.batch_size:
        raise AssertionError("Replay source counts do not sum to batch_size.")
    return counts


class ExternalReplaySampler:
    """Uniform CPU-backed sampler for real or TRACE replay artifacts."""

    def __init__(
        self,
        path: str | Path,
        *,
        seed: int,
        expected_observation_dim: int,
        expected_action_dim: int,
        expected_gamma: float,
        expected_n_step: int,
        expected_source: Literal["real", "sim"],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.sha256 = _sha256_file(self.path)
        artifact = torch.load(self.path, map_location="cpu", weights_only=False)
        format_version = artifact.get("format_version")
        if format_version not in SUPPORTED_EXTERNAL_FORMATS:
            raise ValueError(f"Unsupported replay format {format_version!r}: {self.path}")
        if expected_source == "real" and format_version != "go2_real_replay_v1":
            raise ValueError(f"Expected a real replay artifact, got {format_version!r}.")
        if expected_source == "sim" and not str(format_version).startswith("go2_trace_replay_"):
            raise ValueError(f"Expected a TRACE replay artifact, got {format_version!r}.")

        self.format_version = str(format_version)
        self.metadata = dict(artifact.get("metadata") or {})
        self.data: dict[str, torch.Tensor] = {}
        sizes: set[int] = set()
        for key in REPLAY_KEYS:
            value = artifact.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Replay artifact is missing tensor {key!r}: {self.path}")
            value = value.detach().cpu().contiguous()
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Replay tensor {key!r} contains NaN/Inf: {self.path}")
            self.data[key] = value
            sizes.add(int(value.shape[0]))
        if len(sizes) != 1 or next(iter(sizes), 0) <= 0:
            raise ValueError("Replay tensors must have one shared non-zero leading dimension.")
        self.num_transitions = next(iter(sizes))

        observation_dim = int(self.data["observation"].shape[-1])
        next_observation_dim = int(self.data["next_observation"].shape[-1])
        action_dim = int(self.data["action"].shape[-1])
        if observation_dim != expected_observation_dim or next_observation_dim != expected_observation_dim:
            raise ValueError(
                f"Replay observation dim mismatch: {observation_dim}/{next_observation_dim} "
                f"versus expected {expected_observation_dim}."
            )
        if action_dim != expected_action_dim:
            raise ValueError(f"Replay action dim {action_dim} versus expected {expected_action_dim}.")
        gamma = self.metadata.get("gamma")
        n_step = self.metadata.get("n_step")
        if gamma is None or abs(float(gamma) - float(expected_gamma)) > 1.0e-9:
            raise ValueError(f"Replay gamma {gamma!r} versus expected {expected_gamma}.")
        if n_step is None or int(n_step) != int(expected_n_step):
            raise ValueError(f"Replay n_step {n_step!r} versus expected {expected_n_step}.")

        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def sample(self, count: int, *, device: torch.device | str) -> dict[str, torch.Tensor]:
        if count <= 0:
            raise ValueError("ExternalReplaySampler.sample count must be positive.")
        indices = torch.randint(
            self.num_transitions,
            (int(count),),
            generator=self.generator,
            device="cpu",
        )
        return {key: value[indices].to(device, non_blocking=True) for key, value in self.data.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("External replay sampler state is missing generator_state.")
        self.generator.set_state(generator_state.detach().cpu())


def _sample_rwm_buffer(
    rwm_buffer: Any,
    count: int,
    *,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if count <= 0:
        raise ValueError("RWM sample count must be positive.")
    try:
        size = len(rwm_buffer)
    except TypeError as error:
        raise ValueError("RWM buffer must implement __len__.") from error
    if size <= 0:
        raise ValueError("RWM buffer is empty.")
    indices = torch.randint(size, (int(count),), generator=generator, device="cpu")
    batch = rwm_buffer.sample(sample_idxs=indices)
    if not isinstance(batch, dict):
        raise ValueError("RWM buffer sample must be a dictionary.")
    return batch


def _validate_batch(
    batch: dict[str, torch.Tensor],
    *,
    source: str,
    count: int,
    observation_dim: int,
    action_dim: int,
) -> None:
    for key in REPLAY_KEYS:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{source} batch is missing tensor {key!r}.")
        if int(value.shape[0]) != count:
            raise ValueError(f"{source}.{key} leading dim {value.shape[0]} versus expected {count}.")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{source}.{key} contains NaN/Inf.")
    for key in ("observation", "next_observation"):
        if batch[key].ndim != 2:
            raise ValueError(f"{source}.{key} must have shape [batch, dim].")
        if int(batch[key].shape[-1]) != observation_dim:
            raise ValueError(
                f"{source}.{key} dim {batch[key].shape[-1]} versus expected {observation_dim}."
            )
    if batch["action"].ndim != 2:
        raise ValueError(f"{source}.action must have shape [batch, dim].")
    if int(batch["action"].shape[-1]) != action_dim:
        raise ValueError(f"{source}.action dim {batch['action'].shape[-1]} versus expected {action_dim}.")
    for key in ("reward", "terminated", "truncated"):
        if batch[key].ndim != 1:
            raise ValueError(f"{source}.{key} must have shape [batch], got {tuple(batch[key].shape)}.")
    for key in REPLAY_KEYS:
        if batch[key].dtype != torch.float32:
            raise ValueError(f"{source}.{key} dtype {batch[key].dtype} versus expected torch.float32.")


def _source_stats(prefix: str, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    reward = batch["reward"].float()
    return {
        f"Replay/{prefix}_reward_mean": float(reward.mean().detach().cpu()),
        f"Replay/{prefix}_reward_std": float(reward.std(unbiased=False).detach().cpu()),
        f"Replay/{prefix}_terminated_fraction": float(
            batch["terminated"].float().mean().detach().cpu()
        ),
        f"Replay/{prefix}_truncated_fraction": float(
            batch["truncated"].float().mean().detach().cpu()
        ),
    }


def sample_mixed_replay_batch(
    *,
    config: ReplayMixConfig,
    observation_dim: int,
    action_dim: int,
    device: torch.device | str,
    real_sampler: ExternalReplaySampler | None,
    rwm_buffer: Any | None,
    sim_sampler: ExternalReplaySampler | None,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    """Sample, concatenate, and shuffle one exact-composition update batch."""

    counts = compute_source_counts(config)
    command_region_ratios = getattr(
        sim_sampler, "command_region_sim_ratios", None
    )
    if command_region_ratios is not None:
        if config.synthetic_mode != "mixed":
            raise ValueError(
                "Command-region TRACE injection requires synthetic_mode='mixed'."
            )
        mix_by_region = getattr(
            sim_sampler, "mix_rwm_by_command_region", None
        )
        if not callable(mix_by_region):
            raise TypeError(
                "Command-region TRACE sampler must implement "
                "mix_rwm_by_command_region()."
            )
        if rwm_buffer is None:
            raise ValueError(
                "Command-region TRACE injection requires the RWM buffer."
            )
        real_count = int(config.batch_size * config.real_ratio)
        synthetic_count = config.batch_size - real_count
        if real_count and real_sampler is None:
            raise ValueError(
                "real_sampler is required by the configured real count."
            )
        real_batch = (
            real_sampler.sample(real_count, device=device)
            if real_count
            else None
        )
        rwm_synthetic = _sample_rwm_buffer(
            rwm_buffer, synthetic_count, generator=generator
        )
        _validate_batch(
            rwm_synthetic,
            source="rwm",
            count=synthetic_count,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
        mixed_synthetic, sim_mask_cpu, region_report = mix_by_region(
            rwm_synthetic, generator=generator, device=device
        )
        _validate_batch(
            mixed_synthetic,
            source="command_region_synthetic",
            count=synthetic_count,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
        if tuple(sim_mask_cpu.shape) != (synthetic_count,):
            raise ValueError("Command-region TRACE sampler returned a bad mask.")
        sim_mask = sim_mask_cpu.to(device=device, dtype=torch.bool)
        sim_count = int(sim_mask.sum().item())
        rwm_count = synthetic_count - sim_count
        source_batches: list[dict[str, torch.Tensor]] = []
        source_ids_parts: list[torch.Tensor] = []
        if real_batch is not None:
            _validate_batch(
                real_batch,
                source="real",
                count=real_count,
                observation_dim=observation_dim,
                action_dim=action_dim,
            )
            source_batches.append(
                {key: real_batch[key].to(device) for key in REPLAY_KEYS}
            )
            source_ids_parts.append(
                torch.full(
                    (real_count,), SOURCE_REAL, dtype=torch.int64, device=device
                )
            )
        source_batches.append(mixed_synthetic)
        source_ids_parts.append(
            torch.where(
                sim_mask,
                torch.full(
                    (synthetic_count,),
                    SOURCE_SIM,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (synthetic_count,),
                    SOURCE_RWM,
                    dtype=torch.int64,
                    device=device,
                ),
            )
        )
        mixed = {
            key: torch.cat([batch[key] for batch in source_batches], dim=0)
            for key in REPLAY_KEYS
        }
        source_ids = torch.cat(source_ids_parts)
        if config.shuffle:
            permutation = torch.randperm(
                config.batch_size, generator=generator, device="cpu"
            ).to(device)
            mixed = {key: value[permutation] for key, value in mixed.items()}
            source_ids = source_ids[permutation]
        info: dict[str, float] = {
            "Replay/real_count": float(real_count),
            "Replay/rwm_count": float(rwm_count),
            "Replay/sim_count": float(sim_count),
            "Replay/real_ratio_actual": real_count / config.batch_size,
            "Replay/rwm_ratio_actual": rwm_count / config.batch_size,
            "Replay/sim_ratio_actual": sim_count / config.batch_size,
            "Replay/trace_warmup": float(
                bool(
                    getattr(sim_sampler, "metadata", {}).get(
                        "mutable", False
                    )
                )
                and len(sim_sampler) == 0
            ),
            "Replay/mixed_reward_mean": float(
                mixed["reward"].float().mean().detach().cpu()
            ),
            "Replay/mixed_reward_std": float(
                mixed["reward"].float().std(unbiased=False).detach().cpu()
            ),
        }
        info.update(
            {
                f"Replay/region_{key}": float(value)
                for key, value in region_report.items()
            }
        )
        if real_batch is not None:
            info.update(
                _source_stats(
                    "real",
                    {key: real_batch[key].to(device) for key in REPLAY_KEYS},
                )
            )
        if rwm_count:
            info.update(
                _source_stats(
                    "rwm",
                    {
                        key: mixed_synthetic[key][~sim_mask]
                        for key in REPLAY_KEYS
                    },
                )
            )
        if sim_count:
            info.update(
                _source_stats(
                    "sim",
                    {
                        key: mixed_synthetic[key][sim_mask]
                        for key in REPLAY_KEYS
                    },
                )
            )
        return mixed, info, source_ids

    sim_warmup = bool(
        counts.sim
        and sim_sampler is not None
        and bool(getattr(sim_sampler, "metadata", {}).get("mutable", False))
        and len(sim_sampler) == 0
    )
    real_count = counts.real
    sim_count = 0 if sim_warmup else counts.sim
    rwm_count = counts.rwm + (counts.sim if sim_warmup else 0)
    source_batches: list[tuple[str, int, int, dict[str, torch.Tensor]]] = []
    if real_count:
        if real_sampler is None:
            raise ValueError("real_sampler is required by the configured real count.")
        source_batches.append(
            ("real", SOURCE_REAL, real_count, real_sampler.sample(real_count, device=device))
        )
    if rwm_count:
        if rwm_buffer is None:
            raise ValueError("rwm_buffer is required by the configured RWM count.")
        source_batches.append(
            ("rwm", SOURCE_RWM, rwm_count, _sample_rwm_buffer(rwm_buffer, rwm_count, generator=generator))
        )
    if sim_count:
        if sim_sampler is None:
            raise ValueError("sim_sampler is required by the configured simulator count.")
        source_batches.append(
            ("sim", SOURCE_SIM, sim_count, sim_sampler.sample(sim_count, device=device))
        )

    for name, _source_id, count, batch in source_batches:
        _validate_batch(
            batch,
            source=name,
            count=count,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
    reference_batch = source_batches[0][3]
    for name, _source_id, _count, batch in source_batches[1:]:
        for key in REPLAY_KEYS:
            if tuple(batch[key].shape[1:]) != tuple(reference_batch[key].shape[1:]):
                raise ValueError(
                    f"{name}.{key} trailing shape {tuple(batch[key].shape[1:])} does not match "
                    f"{tuple(reference_batch[key].shape[1:])}."
                )
            if batch[key].dtype != reference_batch[key].dtype:
                raise ValueError(
                    f"{name}.{key} dtype {batch[key].dtype} does not match {reference_batch[key].dtype}."
                )

    mixed = {
        key: torch.cat([batch[key].to(device) for _name, _sid, _count, batch in source_batches], dim=0)
        for key in REPLAY_KEYS
    }
    source_ids = torch.cat(
        [
            torch.full((count,), source_id, dtype=torch.int64, device=device)
            for _name, source_id, count, _batch in source_batches
        ]
    )
    if int(mixed["reward"].shape[0]) != config.batch_size:
        raise AssertionError("Mixed replay batch has the wrong total size.")

    if config.shuffle:
        permutation = torch.randperm(config.batch_size, generator=generator, device="cpu").to(device)
        mixed = {key: value[permutation] for key, value in mixed.items()}
        source_ids = source_ids[permutation]

    info: dict[str, float] = {
        "Replay/real_count": float(real_count),
        "Replay/rwm_count": float(rwm_count),
        "Replay/sim_count": float(sim_count),
        "Replay/real_ratio_actual": real_count / config.batch_size,
        "Replay/rwm_ratio_actual": rwm_count / config.batch_size,
        "Replay/sim_ratio_actual": sim_count / config.batch_size,
        "Replay/trace_warmup": float(sim_warmup),
        "Replay/mixed_reward_mean": float(mixed["reward"].float().mean().detach().cpu()),
        "Replay/mixed_reward_std": float(
            mixed["reward"].float().std(unbiased=False).detach().cpu()
        ),
    }
    for name, _source_id, _count, batch in source_batches:
        info.update(_source_stats(name, batch))
    return mixed, info, source_ids
