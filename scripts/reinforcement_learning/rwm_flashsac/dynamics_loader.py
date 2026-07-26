"""Dynamics checkpoint loader helpers for Go2 FlashSAC-RWM training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scripts.reinforcement_learning.rwm.dynamics import load_dynamics_checkpoint
from scripts.reinforcement_learning.rwm_dataset.proprioceptive_dynamics import (
    load_proprioceptive_dynamics_checkpoint,
)


def load_any_go2_dynamics_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[Any, dict[str, Any]]:
    """Load either the original full-state RWM or the proprioceptive RWM.

    The proprioceptive model consumes ``state[..., 3:45]`` internally, but still
    returns the original full 45-dim next-state prediction. That keeps the
    existing FlashSAC imagination env and 48-dim policy observation compatible.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos") or {}
    model_type = checkpoint.get("model_type") or infos.get("model_type")
    if model_type in {"go2_proprioceptive", "go2_reduced_input_no_base_lin_vel"}:
        return load_proprioceptive_dynamics_checkpoint(path, device=device)
    return load_dynamics_checkpoint(path, device=device)
