"""V13 full-state entry point for the stock model_based Go2 RWM trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset import train_world_model_offline_go2 as baseline
from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key


def _v13_full_policy_obs_stats(
    dataset: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Reconstruct the frozen 48D policy view instead of normalizing legacy 45D obs."""

    states = stack_time_key(dataset, "states").float()
    commands = stack_time_key(dataset, "commands").float()
    prev_actions = stack_time_key(dataset, "prev_actions").float()
    if states.shape[-1] != 45:
        raise ValueError(f"V13 full-state training requires state_dim=45, got {states.shape[-1]}.")
    if commands.shape[-1] != 3:
        raise ValueError(f"V13 full-state training requires command_dim=3, got {commands.shape[-1]}.")
    if prev_actions.shape[-1] != 12:
        raise ValueError(
            f"V13 full-state training requires prev_action_dim=12, got {prev_actions.shape[-1]}."
        )
    observations = torch.cat(
        (states[..., 0:9], commands, states[..., 9:33], prev_actions),
        dim=-1,
    ).reshape(-1, 48)
    if not bool(torch.isfinite(observations).all()):
        raise ValueError("Reconstructed V13 48D policy observations contain NaN/Inf.")
    return {
        "obs_mean": observations.mean(dim=0),
        "obs_std": observations.std(dim=0).clamp_min(1.0e-6),
    }


baseline._obs_stats = _v13_full_policy_obs_stats


if __name__ == "__main__":
    baseline.main()
