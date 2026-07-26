from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.reinforcement_learning.rwm.dynamics import (
    DynamicsConfig,
    SystemDynamicsEnsemble,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "scripts/reinforcement_learning/rwm_dataset/configs/"
    "go2_offline_world_model_v13_full45.yaml"
)


def _small_model() -> SystemDynamicsEnsemble:
    return SystemDynamicsEnsemble(
        DynamicsConfig(
            state_dim=45,
            action_dim=12,
            ensemble_size=2,
            history_horizon=2,
            forecast_horizon=1,
            hidden_size=16,
            num_layers=1,
            loss_mode="reference_autoregressive_mse",
        )
    )


def test_v13_config_is_fullstate_45_to_45() -> None:
    cfg = OmegaConf.load(CONFIG_PATH)
    assert list(cfg.action_mask_indices) == []
    assert int(cfg.v13_interface.rwm_input_state_dim) == 45
    assert int(cfg.v13_interface.rwm_output_state_dim) == 45
    assert int(cfg.v13_interface.actor_observation_dim) == 48
    assert int(cfg.v13_interface.critic_observation_dim) == 48
    assert bool(cfg.v13_interface.base_lin_vel_input_present)
    assert bool(cfg.v13_interface.base_lin_vel_target_supervised)


def test_blv_is_visible_to_input_and_supervised_in_output() -> None:
    torch.manual_seed(123)
    model = _small_model()
    batch_size, sequence_length = 2, 3
    states = torch.randn(batch_size, sequence_length, 45)
    actions = torch.randn(batch_size, sequence_length, 12)
    next_states = torch.randn(batch_size, sequence_length, 45)
    contacts = torch.zeros(batch_size, sequence_length, 4)
    terminations = torch.zeros(batch_size, sequence_length, 1)

    losses = model.compute_loss(
        states,
        actions,
        next_states,
        contacts,
        terminations,
        bootstrap=False,
    )
    losses["total_loss"].backward()
    output_head = model.members[0].state_mean[-1]
    assert float(output_head.weight.grad[:3].abs().sum()) > 0.0

    changed_input_blv = states.clone()
    changed_input_blv[..., :3] += 1000.0
    model.eval()
    model_ids = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        prediction_a = model.predict(
            states[:, :2],
            actions[:, :2],
            model_ids=model_ids,
        )[0]
        prediction_b = model.predict(
            changed_input_blv[:, :2],
            actions[:, :2],
            model_ids=model_ids,
        )[0]
    assert not torch.equal(prediction_a, prediction_b)
