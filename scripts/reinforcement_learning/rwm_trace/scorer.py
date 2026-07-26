"""Per-dataset Bradley-Terry scorer for the formal Go2 TRACE path."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .schemas import (
    PROMPT_HASH,
    SCORER_EXPANDED_FEATURE_NAMES,
    SCORER_FEATURE_NAMES,
    SCORER_FEATURE_SCHEMA_HASH,
    SUMMARY_SCHEMA_HASH,
    canonical_sha256,
)


CHECKPOINT_FORMAT = "go2_trace_scorer_v2"
MODEL_ARCHITECTURE = "mlp_relu_2x256_scalar_v1"


@dataclass(frozen=True)
class FeatureStats:
    base_names: tuple[str, ...]
    imputation_mean: tuple[float, ...]
    expanded_mean: tuple[float, ...]
    expanded_std: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureStats":
        return cls(
            base_names=tuple(map(str, value["base_names"])),
            imputation_mean=tuple(map(float, value["imputation_mean"])),
            expanded_mean=tuple(map(float, value["expanded_mean"])),
            expanded_std=tuple(map(float, value["expanded_std"])),
        )

    def validate(self) -> None:
        width = len(SCORER_FEATURE_NAMES)
        if self.base_names != SCORER_FEATURE_NAMES:
            raise ValueError("Checkpoint scorer feature names do not match the fixed schema.")
        if not (
            len(self.imputation_mean) == width
            and len(self.expanded_mean) == 2 * width
            and len(self.expanded_std) == 2 * width
        ):
            raise ValueError("Checkpoint feature-stat dimensions are inconsistent.")
        if np.any(np.asarray(self.expanded_std) <= 0.0):
            raise ValueError("Checkpoint feature standard deviations must be positive.")


@dataclass(frozen=True)
class ScorerBinding:
    task_id: str
    dataset_id: str
    dataset_sha256: str
    condition_id: str
    labels_sha256: str
    split_sha256: str
    clean_core_commit: str = "de722be372a610f08d004e65b3b4f05db2052b6a"
    clean_scorer_commit: str = "e9235f5143a12f7b31c8b447afee4077007da360"
    summary_schema_hash: str = SUMMARY_SCHEMA_HASH
    feature_schema_hash: str = SCORER_FEATURE_SCHEMA_HASH
    prompt_hash: str = PROMPT_HASH
    model_architecture: str = MODEL_ARCHITECTURE

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if not str(value):
                raise ValueError(f"Scorer binding field {key!r} must be non-empty.")
        if self.summary_schema_hash != SUMMARY_SCHEMA_HASH:
            raise ValueError("Scorer binding summary schema is stale.")
        if self.feature_schema_hash != SCORER_FEATURE_SCHEMA_HASH:
            raise ValueError("Scorer binding feature schema is stale.")
        if self.prompt_hash != PROMPT_HASH:
            raise ValueError("Scorer binding prompt is stale.")
        if self.model_architecture != MODEL_ARCHITECTURE:
            raise ValueError("Scorer binding model architecture is unsupported.")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScorerBinding":
        return cls(**{key: str(item) for key, item in value.items()})


class Go2TraceScorer(nn.Module):
    """Scalar utility model used through score differences in BT training."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if hidden_dim != 256:
            raise ValueError("Formal Go2 TRACE scorer uses exactly two 256-wide hidden layers.")
        if input_dim != len(SCORER_EXPANDED_FEATURE_NAMES):
            raise ValueError("Scorer input width does not match the fixed expanded feature schema.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def raw_feature_matrix(summaries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for summary in summaries:
        if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
            raise ValueError("Scorer received a trajectory summary from another schema.")
        row: list[float] = []
        for name in SCORER_FEATURE_NAMES:
            try:
                value = float(summary.get(name, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            row.append(value if np.isfinite(value) else float("nan"))
        rows.append(row)
    return (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.zeros((0, len(SCORER_FEATURE_NAMES)), dtype=np.float64)
    )


def fit_feature_stats(raw: np.ndarray) -> FeatureStats:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(SCORER_FEATURE_NAMES) or len(raw) == 0:
        raise ValueError("Training features must be non-empty [N, fixed_feature_width].")
    finite = np.isfinite(raw)
    counts = finite.sum(axis=0)
    sums = np.where(finite, raw, 0.0).sum(axis=0)
    imputation = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    imputed = np.where(finite, raw, imputation)
    missing = (~finite).astype(np.float64)
    expanded = np.concatenate((imputed, missing), axis=1)
    mean = expanded.mean(axis=0)
    std = expanded.std(axis=0)
    std[std < 1.0e-6] = 1.0
    return FeatureStats(
        base_names=SCORER_FEATURE_NAMES,
        imputation_mean=tuple(map(float, imputation)),
        expanded_mean=tuple(map(float, mean)),
        expanded_std=tuple(map(float, std)),
    )


def transform_features(raw: np.ndarray, stats: FeatureStats) -> np.ndarray:
    stats.validate()
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(stats.base_names):
        raise ValueError("Raw scorer feature width mismatch.")
    finite = np.isfinite(raw)
    imputed = np.where(finite, raw, np.asarray(stats.imputation_mean))
    expanded = np.concatenate((imputed, (~finite).astype(np.float64)), axis=1)
    normalized = (
        expanded - np.asarray(stats.expanded_mean)
    ) / np.asarray(stats.expanded_std)
    if not np.isfinite(normalized).all():
        raise ValueError("Feature normalization produced non-finite values.")
    return normalized.astype(np.float32)


def feature_matrix(
    summaries: Sequence[Mapping[str, Any]],
    stats: FeatureStats | None = None,
) -> np.ndarray:
    """Compatibility helper: raw fixed features, or transformed if stats is given."""

    raw = raw_feature_matrix(summaries)
    return raw if stats is None else transform_features(raw, stats)


def save_scorer_checkpoint(
    path: str | Path,
    model: Go2TraceScorer,
    stats: FeatureStats,
    *,
    binding: ScorerBinding,
    training_metrics: Mapping[str, Any],
) -> None:
    stats.validate()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "input_dim": len(SCORER_EXPANDED_FEATURE_NAMES),
        "hidden_dim": 256,
        "feature_stats": stats.to_dict(),
        "binding": binding.to_dict(),
        "binding_sha256": binding.sha256,
        "training_metrics": dict(training_metrics),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_scorer_checkpoint(
    path: str | Path,
    *,
    expected_binding: ScorerBinding | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Go2TraceScorer, FeatureStats, ScorerBinding, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("Not a formal Go2 TRACE scorer v2 checkpoint.")
    binding = ScorerBinding.from_dict(checkpoint["binding"])
    if checkpoint.get("binding_sha256") != binding.sha256:
        raise ValueError("Scorer checkpoint binding hash is corrupt.")
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("Scorer checkpoint binding does not match the active task/dataset.")
    if int(checkpoint.get("input_dim", -1)) != len(SCORER_EXPANDED_FEATURE_NAMES):
        raise ValueError("Scorer checkpoint input dimension is incompatible.")
    stats = FeatureStats.from_dict(checkpoint["feature_stats"])
    stats.validate()
    model = Go2TraceScorer(len(SCORER_EXPANDED_FEATURE_NAMES), 256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, stats, binding, dict(checkpoint.get("training_metrics") or {})


def score_summaries(
    model: Go2TraceScorer,
    summaries: Sequence[Mapping[str, Any]],
    stats: FeatureStats,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    features = transform_features(raw_feature_matrix(summaries), stats)
    with torch.no_grad():
        score = model(torch.as_tensor(features, dtype=torch.float32, device=device))
    result = score.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Scorer produced non-finite values.")
    return result
