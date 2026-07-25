"""Helpers for Go2 RWM FlashSAC policies with proprioceptive actor input."""

from __future__ import annotations

import numpy as np
import torch


PROPRIOCEPTIVE_ACTOR_OBS_DIM = 45
CRITIC_OBS_DIM_FULL_RWM = 48


def proprioceptive_obs_np(full_rwm_obs: np.ndarray) -> np.ndarray:
    """Actor view: RWM obs without base_lin_vel[0:3]."""

    return full_rwm_obs[:, 3:].astype(np.float32, copy=False)


def proprioceptive_obs_t(full_rwm_obs: torch.Tensor) -> torch.Tensor:
    """Torch actor view: RWM obs without base_lin_vel[0:3]."""

    return full_rwm_obs[:, 3:]


__all__ = [
    "PROPRIOCEPTIVE_ACTOR_OBS_DIM",
    "CRITIC_OBS_DIM_FULL_RWM",
    "proprioceptive_obs_np",
    "proprioceptive_obs_t",
]
