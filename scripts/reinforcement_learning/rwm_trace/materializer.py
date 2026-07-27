"""Materialize selected trajectories with injected V13 reward/n-step semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch

from .replay import REPLAY_KEYS, MutableTraceReplayBuffer


class V13ReplaySemantics(Protocol):
    """Narrow adapter owned by the V13 FlashSAC/RWM training implementation."""

    @property
    def gamma(self) -> float: ...

    @property
    def n_step(self) -> int: ...

    @property
    def reward_config_sha256(self) -> str: ...

    def materialize(self, trajectory: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """Apply the exact V13 public reward and n-step replay semantics."""


@dataclass(frozen=True)
class MaterializationReport:
    trajectory_count: int
    transition_count: int
    gamma: float
    n_step: int
    reward_config_sha256: str


def materialize_selected(
    trajectories: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    scores: Sequence[float],
    *,
    semantics: V13ReplaySemantics,
    buffer: MutableTraceReplayBuffer,
    refresh_step: int,
    score_source: str = "learned",
) -> MaterializationReport:
    """Insert selected candidates; never implement a reward fallback locally."""

    if not semantics.reward_config_sha256:
        raise ValueError("V13 reward adapter must expose a non-empty config hash.")
    if not 0.0 < float(semantics.gamma) <= 1.0 or int(semantics.n_step) < 1:
        raise ValueError("V13 replay adapter exposes invalid gamma/n_step.")
    seen: set[int] = set()
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for raw_index in selected_indices:
        index = int(raw_index)
        if index in seen or not 0 <= index < len(trajectories):
            raise ValueError("Selected trajectory indices contain duplicate/out-of-range values.")
        seen.add(index)
        selected.append((index, trajectories[index]))

    materialize_many = getattr(semantics, "materialize_many", None)
    if callable(materialize_many):
        replay_batches = list(
            materialize_many([trajectory for _index, trajectory in selected])
        )
    else:
        replay_batches = [
            semantics.materialize(trajectory)
            for _index, trajectory in selected
        ]
    if len(replay_batches) != len(selected):
        raise ValueError("V13 batch materializer returned the wrong batch count.")
    for replay in replay_batches:
        if set(replay) != set(REPLAY_KEYS):
            raise ValueError("V13 materializer must return exactly the six replay tensors.")

    inserted = buffer.add_batches(
        replay_batches,
        trajectory_ids=[
            str(trajectory.get("trajectory_id", index))
            for index, trajectory in selected
        ],
        scores=[float(scores[index]) for index, _trajectory in selected],
        refresh_step=refresh_step,
        score_source=score_source,
    )
    return MaterializationReport(
        trajectory_count=len(seen),
        transition_count=inserted,
        gamma=float(semantics.gamma),
        n_step=int(semantics.n_step),
        reward_config_sha256=str(semantics.reward_config_sha256),
    )
