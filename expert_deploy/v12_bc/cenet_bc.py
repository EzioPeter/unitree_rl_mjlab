"""CENet-augmented behavior-cloning models for the V12 Go2 datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from flash_rl.agents.flashSAC.network import FlashSACActor


@dataclass(frozen=True)
class CENetBCModelConfig:
    observation_dim: int = 45
    action_dim: int = 12
    history_length: int = 10
    cenet_hidden_dim: int = 128
    latent_dim: int = 16
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2

    @property
    def actor_input_dim(self) -> int:
        return self.observation_dim + 3 + self.latent_dim


class CENet(nn.Module):
    """Estimate body-frame base velocity and a context latent from observation history."""

    def __init__(
        self,
        *,
        history_length: int,
        observation_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.observation_dim = int(observation_dim)
        input_dim = self.history_length * self.observation_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.velocity_head = nn.Linear(hidden_dim, 3)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, observation_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observation_history.ndim != 3:
            raise ValueError(
                "CENet expects observation history [batch, history, observation], "
                f"got {tuple(observation_history.shape)}."
            )
        x = observation_history.flatten(start_dim=1)
        x = self.encoder(self.input_norm(x))
        return self.velocity_head(x), torch.tanh(self.latent_head(x))


class CENetBCPolicy(nn.Module):
    """Joint CENet and deterministic FlashSAC actor used during training."""

    def __init__(self, config: CENetBCModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cenet = CENet(
            history_length=config.history_length,
            observation_dim=config.observation_dim,
            hidden_dim=config.cenet_hidden_dim,
            latent_dim=config.latent_dim,
        )
        self.actor = FlashSACActor(
            num_blocks=config.actor_num_blocks,
            input_dim=config.actor_input_dim,
            hidden_dim=config.actor_hidden_dim,
            action_dim=config.action_dim,
        )

    def predict(
        self,
        observation_history: torch.Tensor,
        observation: torch.Tensor,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_linear_velocity, latent = self.cenet(observation_history)
        actor_input = torch.cat((observation, base_linear_velocity, latent), dim=-1)
        mean, _std = self.actor.get_mean_and_std(actor_input, training=training)
        return torch.tanh(mean), base_linear_velocity


class CENetBCDeploymentPolicy(nn.Module):
    """Exportable deterministic policy returning actions and estimated base velocity."""

    def __init__(self, policy: CENetBCPolicy) -> None:
        super().__init__()
        self.cenet = policy.cenet
        self.actor = policy.actor

    def forward(
        self,
        observation_history: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_linear_velocity, latent = self.cenet(observation_history)
        x = torch.cat((observation, base_linear_velocity, latent), dim=-1)
        x = self.actor.embedder(x, training=False)
        for block in self.actor.encoder:
            x = block(x, training=False)
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.actor.post_norm.eps)
        x = x * self.actor.post_norm.weight
        mean = F.linear(
            x,
            self.actor.predictor.mean_w.w.weight,
            self.actor.predictor.mean_bias,
        )
        return torch.tanh(mean), base_linear_velocity


def normalize_unit_parameters(module: nn.Module) -> None:
    """Apply the unit-norm constraints used by FlashSAC layers."""

    with torch.no_grad():
        for child in module.modules():
            normalize = getattr(child, "normalize_parameters", None)
            if callable(normalize):
                normalize()

