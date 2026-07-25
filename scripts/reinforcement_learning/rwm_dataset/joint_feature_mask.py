"""Feature-index helpers for Go2 joint-mask ablation experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch

from scripts.reinforcement_learning.rwm_dataset.action_mask import normalize_action_mask_indices
from scripts.reinforcement_learning.rwm_dataset.broken_go2 import GO2_ACTION_JOINT_NAMES


GO2_RWM_STATE_DIM = 45
GO2_FULL_POLICY_OBS_DIM = 48
GO2_BASE_LIN_VEL_STATE_INDICES: tuple[int, ...] = (0, 1, 2)

GO2_RWM_STATE_JOINT_FIELD_OFFSETS: dict[str, int] = {
    "joint_pos": 9,
    "joint_vel": 21,
    "actuator_force": 33,
}

GO2_FULL_POLICY_OBS_JOINT_FIELD_OFFSETS: dict[str, int] = {
    "joint_pos": 12,
    "joint_vel": 24,
    "last_action": 36,
}


def indices_to_keep(
    dropped_indices: int | str | Iterable[int | str] | None,
    *,
    dim: int,
) -> tuple[int, ...]:
    dropped = set(normalize_action_mask_indices(dropped_indices, action_dim=dim))
    return tuple(idx for idx in range(int(dim)) if idx not in dropped)


def _joint_name_to_index(joint_name: str) -> int:
    try:
        return GO2_ACTION_JOINT_NAMES.index(str(joint_name))
    except ValueError as exc:
        raise ValueError(
            f"Unknown Go2 joint name {joint_name!r}. Known joints: {list(GO2_ACTION_JOINT_NAMES)}"
        ) from exc


def go2_joint_names_to_rwm_state_indices(
    joint_names: Sequence[str],
    *,
    include_actuator_force: bool = True,
) -> tuple[int, ...]:
    fields = ("joint_pos", "joint_vel", "actuator_force") if include_actuator_force else ("joint_pos", "joint_vel")
    indices: list[int] = []
    for joint_name in dict.fromkeys(str(name) for name in joint_names if str(name)):
        joint_idx = _joint_name_to_index(joint_name)
        indices.extend(GO2_RWM_STATE_JOINT_FIELD_OFFSETS[field] + joint_idx for field in fields)
    return tuple(dict.fromkeys(indices))


def go2_joint_names_to_policy_obs_indices(joint_names: Sequence[str]) -> tuple[int, ...]:
    indices: list[int] = []
    for joint_name in dict.fromkeys(str(name) for name in joint_names if str(name)):
        joint_idx = _joint_name_to_index(joint_name)
        indices.extend(offset + joint_idx for offset in GO2_FULL_POLICY_OBS_JOINT_FIELD_OFFSETS.values())
    return tuple(dict.fromkeys(indices))


def mask_tensor_features_t(
    values: torch.Tensor,
    dropped_indices: int | str | Iterable[int | str] | None,
) -> torch.Tensor:
    keep = indices_to_keep(dropped_indices, dim=int(values.shape[-1]))
    if len(keep) == int(values.shape[-1]):
        return values
    keep_t = torch.tensor(keep, dtype=torch.long, device=values.device)
    return values.index_select(dim=-1, index=keep_t)


def mask_array_features_np(
    values: np.ndarray,
    dropped_indices: int | str | Iterable[int | str] | None,
) -> np.ndarray:
    keep = indices_to_keep(dropped_indices, dim=int(values.shape[-1]))
    if len(keep) == int(values.shape[-1]):
        return values.astype(np.float32, copy=False)
    return values[..., list(keep)].astype(np.float32, copy=False)


def expand_tensor_features_t(
    values: torch.Tensor,
    dropped_indices: int | str | Iterable[int | str] | None,
    *,
    full_dim: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    keep = indices_to_keep(dropped_indices, dim=int(full_dim))
    if len(keep) == int(full_dim):
        return values
    if int(values.shape[-1]) != len(keep):
        raise ValueError(f"Expected reduced dim {len(keep)}, got {int(values.shape[-1])}.")
    out = torch.full(
        (*values.shape[:-1], int(full_dim)),
        float(fill_value),
        dtype=values.dtype,
        device=values.device,
    )
    out[..., list(keep)] = values
    return out

