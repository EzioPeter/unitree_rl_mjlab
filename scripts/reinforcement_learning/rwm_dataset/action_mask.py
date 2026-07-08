"""Action masking helpers for Go2 RWM ablation experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def normalize_action_mask_indices(
    indices: int | str | Iterable[int | str] | None,
    *,
    action_dim: int | None = None,
) -> tuple[int, ...]:
    """Return sorted unique action indices and validate optional bounds."""

    if indices is None:
        values: list[int] = []
    elif isinstance(indices, int):
        values = [indices]
    elif isinstance(indices, str):
        raw = indices.replace(",", " ").split()
        values = [int(item) for item in raw]
    else:
        values = [int(item) for item in indices]

    out = tuple(sorted(set(values)))
    if action_dim is not None:
        bad = [idx for idx in out if idx < 0 or idx >= int(action_dim)]
        if bad:
            raise ValueError(f"Action mask indices out of range for action_dim={action_dim}: {bad}")
    return out


def mask_action_tensor(
    actions: torch.Tensor,
    indices: tuple[int, ...],
    *,
    clone: bool = True,
) -> torch.Tensor:
    """Zero selected indices in the last dimension of an action tensor."""

    if not indices:
        return actions.clone() if clone else actions
    out = actions.clone() if clone else actions
    out[..., list(indices)] = 0.0
    return out


def mask_dataset_actions(
    dataset: dict[str, Any],
    indices: int | str | Iterable[int | str] | None,
) -> tuple[int, ...]:
    """Zero selected action dimensions in a loaded mixed dataset in place."""

    actions = dataset.get("actions")
    if actions is None:
        raise KeyError("Dataset is missing required key 'actions'.")
    if isinstance(actions, torch.Tensor):
        action_dim = int(actions.shape[-1])
    elif isinstance(actions, list) and actions:
        action_dim = int(actions[0].shape[-1])
    else:
        raise ValueError("Dataset key 'actions' must be a non-empty list or tensor.")

    mask_indices = normalize_action_mask_indices(indices, action_dim=action_dim)
    if not mask_indices:
        return ()

    if isinstance(actions, torch.Tensor):
        dataset["actions"] = mask_action_tensor(actions.float(), mask_indices, clone=True)
    else:
        dataset["actions"] = [mask_action_tensor(item.float(), mask_indices, clone=True) for item in actions]

    metadata = dict(dataset.get("metadata") or {})
    metadata["world_model_action_mask_indices"] = list(mask_indices)
    dataset["metadata"] = metadata
    return mask_indices
