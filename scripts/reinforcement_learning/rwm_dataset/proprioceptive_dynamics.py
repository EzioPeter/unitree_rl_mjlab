"""Proprioceptive Go2 dynamics for real-world sensor ablations.

The model consumes only the proprioceptive portion of the Go2 RWM state:

    input_state = full_state[..., 3:45]  # 42 dims

The prediction target/output remains the original full 45-dim next state.
This is different from masking a 45-dim input; the GRU never receives the
three base linear velocity channels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ProprioceptiveDynamicsConfig:
    input_state_dim: int = 42
    output_state_dim: int = 45
    action_dim: int = 12
    contact_dim: int = 4
    termination_dim: int = 1
    ensemble_size: int = 5
    history_horizon: int = 32
    forecast_horizon: int = 8
    hidden_size: int = 256
    num_layers: int = 2
    min_logstd: float = -5.0
    max_logstd: float = 2.0
    state_loss_weight: float = 1.0
    sequence_loss_weight: float = 1.0
    bound_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    extension_loss_weight: float = 1.0
    contact_loss_weight: float = 1.0
    termination_loss_weight: float = 1.0
    loss_mode: str = "reference_autoregressive_mse"
    dropped_state_indices: tuple[int, ...] = (0, 1, 2)


class _ProprioceptiveDynamicsMember(nn.Module):
    def __init__(self, cfg: ProprioceptiveDynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.input_state_dim + cfg.action_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.state_mean = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.output_state_dim),
        )
        self.state_logstd = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.output_state_dim),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.contact_dim),
        )
        self.termination_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.termination_dim),
        )

    def _decode(
        self,
        hidden_output: torch.Tensor,
        normalized_output_reference: torch.Tensor,
        return_raw_logstd: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        delta = self.state_mean(hidden_output)
        mean = normalized_output_reference + delta
        raw_logstd = self.state_logstd(hidden_output)
        logstd = raw_logstd.clamp(self.cfg.min_logstd, self.cfg.max_logstd)
        contact = self.contact_head(hidden_output)
        termination = self.termination_head(hidden_output)
        if return_raw_logstd:
            return mean, logstd, raw_logstd, contact, termination
        return mean, logstd, contact, termination

    def forward(
        self,
        input_state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        normalized_output_reference: torch.Tensor,
        return_raw_logstd: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        x = torch.cat([input_state_hist, action_hist], dim=-1)
        out, _ = self.gru(x)
        return self._decode(out[:, -1], normalized_output_reference, return_raw_logstd=return_raw_logstd)

    def forward_with_hidden(
        self,
        input_state_step: torch.Tensor,
        action_step: torch.Tensor,
        normalized_output_reference: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_raw_logstd: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        x = torch.cat([input_state_step, action_step], dim=-1)
        out, hidden = self.gru(x, hidden)
        outputs = self._decode(out[:, -1], normalized_output_reference, return_raw_logstd=return_raw_logstd)
        return (*outputs, hidden)


class ProprioceptiveSystemDynamicsEnsemble(nn.Module):
    """Ensemble that consumes 42-dim state histories and outputs 45-dim states."""

    def __init__(self, cfg: ProprioceptiveDynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.members = nn.ModuleList([_ProprioceptiveDynamicsMember(cfg) for _ in range(cfg.ensemble_size)])
        self.register_buffer("input_state_mean", torch.zeros(cfg.input_state_dim))
        self.register_buffer("input_state_std", torch.ones(cfg.input_state_dim))
        self.register_buffer("output_state_mean", torch.zeros(cfg.output_state_dim))
        self.register_buffer("output_state_std", torch.ones(cfg.output_state_dim))
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))

    @property
    def ensemble_size(self) -> int:
        return self.cfg.ensemble_size

    def set_normalizers(
        self,
        output_state_mean: torch.Tensor,
        output_state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.output_state_mean.copy_(output_state_mean.to(self.output_state_mean.device))
        self.output_state_std.copy_(output_state_std.to(self.output_state_std.device).clamp_min(1e-6))
        self.input_state_mean.copy_(output_state_mean[3:].to(self.input_state_mean.device))
        self.input_state_std.copy_(output_state_std[3:].to(self.input_state_std.device).clamp_min(1e-6))
        self.action_mean.copy_(action_mean.to(self.action_mean.device))
        self.action_std.copy_(action_std.to(self.action_std.device).clamp_min(1e-6))

    def reduce_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] == self.cfg.input_state_dim:
            return state
        if state.shape[-1] != self.cfg.output_state_dim:
            raise ValueError(
                f"Expected state last dim {self.cfg.input_state_dim} or {self.cfg.output_state_dim}, "
                f"got {state.shape[-1]}."
            )
        return state[..., 3:]

    def normalize_input_state(self, input_state: torch.Tensor) -> torch.Tensor:
        return (input_state - self.input_state_mean) / self.input_state_std

    def normalize_output_state(self, output_state: torch.Tensor) -> torch.Tensor:
        return (output_state - self.output_state_mean) / self.output_state_std

    def denormalize_output_state(self, output_state: torch.Tensor) -> torch.Tensor:
        return output_state * self.output_state_std + self.output_state_mean

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def _normalized_output_reference(self, input_state: torch.Tensor) -> torch.Tensor:
        """Build a 45-d normalized residual reference from a 42-d raw input state."""

        ref = torch.zeros(*input_state.shape[:-1], self.cfg.output_state_dim, device=input_state.device, dtype=input_state.dtype)
        ref[..., 3:] = self.normalize_input_state(input_state)
        return ref

    def _member_prediction(
        self,
        member: _ProprioceptiveDynamicsMember,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        return_raw_logstd: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        input_hist = self.reduce_state(state_hist)
        norm_input_hist = self.normalize_input_state(input_hist)
        norm_action_hist = self.normalize_action(action_hist)
        reference = self._normalized_output_reference(input_hist[:, -1])
        return member(
            norm_input_hist,
            norm_action_hist,
            reference,
            return_raw_logstd=return_raw_logstd,
        )

    def _member_prediction_with_hidden(
        self,
        member: _ProprioceptiveDynamicsMember,
        input_state_step: torch.Tensor,
        action_step: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_raw_logstd: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        input_state_step = self.reduce_state(input_state_step)
        norm_input_step = self.normalize_input_state(input_state_step)
        norm_action_step = self.normalize_action(action_step)
        reference = self._normalized_output_reference(input_state_step[:, -1])
        return member.forward_with_hidden(
            norm_input_step,
            norm_action_step,
            reference,
            hidden=hidden,
            return_raw_logstd=return_raw_logstd,
        )

    def _compute_bound_loss(self, raw_logstd: torch.Tensor) -> torch.Tensor:
        upper = F.relu(raw_logstd - self.cfg.max_logstd).square()
        lower = F.relu(self.cfg.min_logstd - raw_logstd).square()
        return upper.mean() + lower.mean()

    def _compute_reference_autoregressive_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        contacts: torch.Tensor,
        terminations: torch.Tensor,
        bootstrap: bool,
    ) -> dict[str, torch.Tensor]:
        h = self.cfg.history_horizon
        forecast_horizon = min(self.cfg.forecast_horizon, states.shape[1] - h)
        state_losses = []
        sequence_losses = []
        bound_losses = []
        contact_losses = []
        termination_losses = []

        for member in self.members:
            batch_indices = torch.arange(states.shape[0], device=states.device)
            if bootstrap and states.shape[0] > 1:
                batch_indices = torch.randint(0, states.shape[0], (states.shape[0],), device=states.device)

            s = states[batch_indices]
            a = actions[batch_indices]
            ns = next_states[batch_indices]
            c = contacts[batch_indices]
            t = terminations[batch_indices]

            input_state = self.reduce_state(s[:, :h])
            action_input = a[:, :h]
            mean, logstd, raw_logstd, contact_logits, term_logits, hidden = self._member_prediction_with_hidden(
                member,
                input_state,
                action_input,
                hidden=None,
                return_raw_logstd=True,
            )
            state_loss_member = 0.0
            bound_loss_member = 0.0
            contact_loss_member = 0.0
            termination_loss_member = 0.0
            for forecast_idx in range(forecast_horizon):
                if forecast_idx > 0:
                    action_start = h + forecast_idx - 1
                    action_input = a[:, action_start : action_start + 1]
                    mean, logstd, raw_logstd, contact_logits, term_logits, hidden = self._member_prediction_with_hidden(
                        member,
                        input_state,
                        action_input,
                        hidden=hidden,
                        return_raw_logstd=True,
                    )
                target_idx = h + forecast_idx - 1
                target = self.normalize_output_state(ns[:, target_idx])
                pred_sample = torch.randn_like(mean) * torch.exp(logstd) + mean
                state_loss_member = state_loss_member + torch.sum((pred_sample - target).square(), dim=-1).mean()
                bound_loss_member = bound_loss_member + self._compute_bound_loss(raw_logstd)
                contact_loss_member = contact_loss_member + F.binary_cross_entropy_with_logits(
                    contact_logits,
                    c[:, target_idx],
                )
                termination_loss_member = termination_loss_member + F.binary_cross_entropy_with_logits(
                    term_logits,
                    t[:, target_idx],
                )
                input_state = self.reduce_state(self.denormalize_output_state(pred_sample)).unsqueeze(1)

            denom = max(1, forecast_horizon)
            state_losses.append(state_loss_member / denom)
            sequence_losses.append(torch.tensor(0.0, device=states.device))
            bound_losses.append(bound_loss_member / denom)
            contact_losses.append(contact_loss_member / denom)
            termination_losses.append(termination_loss_member / denom)

        zero = torch.tensor(0.0, device=states.device)
        return {
            "state_loss": torch.stack(state_losses).mean(),
            "sequence_loss": torch.stack(sequence_losses).mean(),
            "bound_loss": torch.stack(bound_losses).mean(),
            "kl_loss": zero,
            "extension_loss": zero,
            "contact_loss": torch.stack(contact_losses).mean(),
            "termination_loss": torch.stack(termination_losses).mean(),
        }

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        contacts: torch.Tensor,
        terminations: torch.Tensor,
        bootstrap: bool = True,
    ) -> dict[str, torch.Tensor]:
        losses = self._compute_reference_autoregressive_loss(states, actions, next_states, contacts, terminations, bootstrap)
        total_loss = (
            self.cfg.state_loss_weight * losses["state_loss"]
            + self.cfg.sequence_loss_weight * losses["sequence_loss"]
            + self.cfg.bound_loss_weight * losses["bound_loss"]
            + self.cfg.kl_loss_weight * losses["kl_loss"]
            + self.cfg.extension_loss_weight * losses["extension_loss"]
            + self.cfg.contact_loss_weight * losses["contact_loss"]
            + self.cfg.termination_loss_weight * losses["termination_loss"]
        )
        return {"total_loss": total_loss, **losses}

    @torch.no_grad()
    def predict(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        model_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = []
        logstds = []
        contacts = []
        terms = []
        for member in self.members:
            mean, logstd, contact, term = self._member_prediction(member, state_hist, action_hist)
            means.append(mean)
            logstds.append(logstd)
            contacts.append(contact)
            terms.append(term)
        mean_stack = torch.stack(means, dim=0)
        logstd_stack = torch.stack(logstds, dim=0)
        contact_stack = torch.stack(contacts, dim=0)
        term_stack = torch.stack(terms, dim=0)

        if model_ids is None:
            model_ids = torch.randint(0, self.cfg.ensemble_size, (state_hist.shape[0],), device=state_hist.device)
        gather_idx = model_ids.view(1, -1, 1)
        chosen_mean = mean_stack.gather(0, gather_idx.expand(1, -1, self.cfg.output_state_dim))[0]
        chosen_contact = contact_stack.gather(0, gather_idx.expand(1, -1, self.cfg.contact_dim))[0]
        chosen_term = term_stack.gather(0, gather_idx.expand(1, -1, self.cfg.termination_dim))[0]

        raw_members = self.denormalize_output_state(mean_stack)
        epistemic = raw_members.std(dim=0).mean(dim=-1)
        aleatoric = torch.exp(logstd_stack).mean(dim=(0, 2))
        next_state = self.denormalize_output_state(chosen_mean)
        return next_state, aleatoric, epistemic, chosen_contact, chosen_term

    def checkpoint(
        self,
        optimizer: torch.optim.Optimizer | None,
        iteration: int,
        infos: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "model_type": "go2_proprioceptive",
            "system_dynamics_state_dict": self.state_dict(),
            "system_dynamics_optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "iter": iteration,
            "infos": {
                **(infos or {}),
                "model_type": "go2_proprioceptive",
                "proprioceptive_dynamics_config": asdict(self.cfg),
                "input_state_layout": "full_state[..., 3:45]",
                "output_state_layout": "full 45-dim Go2 RWM state",
            },
        }


def load_proprioceptive_dynamics_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[ProprioceptiveSystemDynamicsEnsemble, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    infos = checkpoint.get("infos") or {}
    cfg_dict = infos.get("proprioceptive_dynamics_config") or infos.get("reduced_input_dynamics_config")
    if cfg_dict is None:
        raise ValueError(
            f"Checkpoint {path} does not contain infos['proprioceptive_dynamics_config']."
        )
    if isinstance(cfg_dict.get("dropped_state_indices"), list):
        cfg_dict["dropped_state_indices"] = tuple(cfg_dict["dropped_state_indices"])
    cfg = ProprioceptiveDynamicsConfig(**cfg_dict)
    dynamics = ProprioceptiveSystemDynamicsEnsemble(cfg).to(device)
    dynamics.load_state_dict(checkpoint["system_dynamics_state_dict"])
    dynamics.eval()
    return dynamics, checkpoint
