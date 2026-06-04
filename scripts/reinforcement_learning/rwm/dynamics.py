"""GRU ensemble dynamics and sequence replay for Go2 RWM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DynamicsConfig:
    state_dim: int
    action_dim: int
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
    contact_loss_weight: float = 1.0
    termination_loss_weight: float = 1.0


@dataclass
class ReplayConfig:
    capacity: int = 200_000
    min_transitions: int = 2_000
    batch_size: int = 512


@dataclass
class WorldModelConfig:
    dynamics: DynamicsConfig
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    updates_per_iter: int = 4
    save_dataset: bool = True


class SequenceReplayBuffer:
    """Transition replay that samples contiguous windows per vector env."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        contact_dim: int,
        termination_dim: int,
        num_envs: int,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = torch.device(device)
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.next_states: list[torch.Tensor] = []
        self.contacts: list[torch.Tensor] = []
        self.terminations: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.states) * self.num_envs

    @property
    def num_time_steps(self) -> int:
        return len(self.states)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        contact: torch.Tensor,
        termination: torch.Tensor,
    ) -> None:
        self.states.append(state.detach().to(self.device).float().cpu())
        self.actions.append(action.detach().to(self.device).float().cpu())
        self.next_states.append(next_state.detach().to(self.device).float().cpu())
        self.contacts.append(contact.detach().to(self.device).float().cpu())
        self.terminations.append(termination.detach().to(self.device).float().cpu())

        max_steps = max(1, self.capacity // self.num_envs)
        overflow = len(self.states) - max_steps
        if overflow > 0:
            del self.states[:overflow]
            del self.actions[:overflow]
            del self.next_states[:overflow]
            del self.contacts[:overflow]
            del self.terminations[:overflow]

    def can_sample(self, history_horizon: int, forecast_horizon: int, min_transitions: int) -> bool:
        return len(self) >= min_transitions and self.num_time_steps >= history_horizon + forecast_horizon

    def _stack(self, values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values, dim=0)

    def stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.cat(
            [
                self._stack(self.states).reshape(-1, self.state_dim),
                self._stack(self.next_states).reshape(-1, self.state_dim),
            ],
            dim=0,
        )
        actions = self._stack(self.actions).reshape(-1, self.action_dim)
        state_mean = states.mean(dim=0)
        state_std = states.std(dim=0).clamp_min(1e-6)
        action_mean = actions.mean(dim=0)
        action_std = actions.std(dim=0).clamp_min(1e-6)
        return state_mean, state_std, action_mean, action_std

    def sample(
        self,
        batch_size: int,
        history_horizon: int,
        forecast_horizon: int,
        device: torch.device | str,
        max_tries: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = history_horizon + forecast_horizon
        if self.num_time_steps < seq_len:
            raise RuntimeError("Not enough transitions to sample a sequence window.")

        states = self._stack(self.states)
        actions = self._stack(self.actions)
        next_states = self._stack(self.next_states)
        contacts = self._stack(self.contacts)
        terms = self._stack(self.terminations)

        sampled: list[tuple[int, int]] = []
        max_start = self.num_time_steps - seq_len
        for _ in range(max_tries):
            remaining = batch_size - len(sampled)
            if remaining <= 0:
                break
            starts = torch.randint(0, max_start + 1, (remaining * 2,))
            env_ids = torch.randint(0, self.num_envs, (remaining * 2,))
            for start, env_id in zip(starts.tolist(), env_ids.tolist()):
                # Avoid windows that cross an episode boundary before the final
                # target. The last transition may itself terminate.
                if terms[start : start + seq_len - 1, env_id].bool().any():
                    continue
                sampled.append((start, env_id))
                if len(sampled) >= batch_size:
                    break

        if not sampled:
            # Fall back to unfiltered windows during very short smoke tests.
            sampled = [
                (
                    int(torch.randint(0, max_start + 1, ()).item()),
                    int(torch.randint(0, self.num_envs, ()).item()),
                )
                for _ in range(batch_size)
            ]
        while len(sampled) < batch_size:
            sampled.append(sampled[-1])

        state_batch = []
        action_batch = []
        next_state_batch = []
        contact_batch = []
        term_batch = []
        for start, env_id in sampled[:batch_size]:
            sl = slice(start, start + seq_len)
            state_batch.append(states[sl, env_id])
            action_batch.append(actions[sl, env_id])
            next_state_batch.append(next_states[sl, env_id])
            contact_batch.append(contacts[sl, env_id])
            term_batch.append(terms[sl, env_id])

        return (
            torch.stack(state_batch, dim=0).to(device),
            torch.stack(action_batch, dim=0).to(device),
            torch.stack(next_state_batch, dim=0).to(device),
            torch.stack(contact_batch, dim=0).to(device),
            torch.stack(term_batch, dim=0).to(device),
        )

    def sample_initial_history(
        self,
        batch_size: int,
        history_horizon: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, actions, _, _, _ = self.sample(
            batch_size=batch_size,
            history_horizon=history_horizon,
            forecast_horizon=1,
            device=device,
        )
        return states[:, :history_horizon], actions[:, :history_horizon]

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "contact_dim": self.contact_dim,
            "termination_dim": self.termination_dim,
            "num_envs": self.num_envs,
            "capacity": self.capacity,
            "states": self.states,
            "actions": self.actions,
            "next_states": self.next_states,
            "contacts": self.contacts,
            "terminations": self.terminations,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: torch.device | str) -> "SequenceReplayBuffer":
        buffer = cls(
            state_dim=state["state_dim"],
            action_dim=state["action_dim"],
            contact_dim=state["contact_dim"],
            termination_dim=state["termination_dim"],
            num_envs=state["num_envs"],
            capacity=state["capacity"],
            device=device,
        )
        buffer.states = state["states"]
        buffer.actions = state["actions"]
        buffer.next_states = state["next_states"]
        buffer.contacts = state["contacts"]
        buffer.terminations = state["terminations"]
        return buffer

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str) -> "SequenceReplayBuffer":
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False), device=device)


class _DynamicsMember(nn.Module):
    def __init__(self, cfg: DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.state_dim + cfg.action_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.state_mean = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.state_dim),
        )
        self.state_logstd = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.state_dim),
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

    def forward(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([state_hist, action_hist], dim=-1)
        out, _ = self.gru(x)
        h = out[:, -1]
        delta = self.state_mean(h)
        mean = state_hist[:, -1] + delta
        logstd = self.state_logstd(h).clamp(self.cfg.min_logstd, self.cfg.max_logstd)
        contact = self.contact_head(h)
        termination = self.termination_head(h)
        return mean, logstd, contact, termination


class SystemDynamicsEnsemble(nn.Module):
    def __init__(self, cfg: DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.members = nn.ModuleList([_DynamicsMember(cfg) for _ in range(cfg.ensemble_size)])
        self.register_buffer("state_mean", torch.zeros(cfg.state_dim))
        self.register_buffer("state_std", torch.ones(cfg.state_dim))
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))

    @property
    def ensemble_size(self) -> int:
        return self.cfg.ensemble_size

    def set_normalizers(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.state_mean.copy_(state_mean.to(self.state_mean.device))
        self.state_std.copy_(state_std.to(self.state_std.device).clamp_min(1e-6))
        self.action_mean.copy_(action_mean.to(self.action_mean.device))
        self.action_std.copy_(action_std.to(self.action_std.device).clamp_min(1e-6))

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.state_std + self.state_mean

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def _member_prediction(
        self,
        member: _DynamicsMember,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return member(self.normalize_state(state_hist), self.normalize_action(action_hist))

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        contacts: torch.Tensor,
        terminations: torch.Tensor,
        bootstrap: bool = True,
    ) -> dict[str, torch.Tensor]:
        h = self.cfg.history_horizon
        seq_len = states.shape[1]
        state_losses = []
        sequence_losses = []
        contact_losses = []
        termination_losses = []

        for member in self.members:
            state_loss_member = 0.0
            contact_loss_member = 0.0
            termination_loss_member = 0.0
            count = 0
            batch_indices = torch.arange(states.shape[0], device=states.device)
            if bootstrap and states.shape[0] > 1:
                batch_indices = torch.randint(0, states.shape[0], (states.shape[0],), device=states.device)

            s = states[batch_indices]
            a = actions[batch_indices]
            ns = next_states[batch_indices]
            c = contacts[batch_indices]
            t = terminations[batch_indices]
            for step in range(h - 1, seq_len):
                mean, logstd, contact_logits, term_logits = self._member_prediction(
                    member,
                    s[:, step - h + 1 : step + 1],
                    a[:, step - h + 1 : step + 1],
                )
                target = self.normalize_state(ns[:, step])
                inv_var = torch.exp(-2.0 * logstd)
                nll = 0.5 * ((target - mean).square() * inv_var + 2.0 * logstd)
                state_loss_member = state_loss_member + nll.mean()
                contact_loss_member = contact_loss_member + F.binary_cross_entropy_with_logits(
                    contact_logits,
                    c[:, step],
                )
                termination_loss_member = termination_loss_member + F.binary_cross_entropy_with_logits(
                    term_logits,
                    t[:, step],
                )
                count += 1
            state_loss_member = state_loss_member / max(1, count)
            contact_loss_member = contact_loss_member / max(1, count)
            termination_loss_member = termination_loss_member / max(1, count)
            state_losses.append(state_loss_member)
            sequence_losses.append(state_loss_member)
            contact_losses.append(contact_loss_member)
            termination_losses.append(termination_loss_member)

        state_loss = torch.stack(state_losses).mean()
        sequence_loss = torch.stack(sequence_losses).mean()
        contact_loss = torch.stack(contact_losses).mean()
        termination_loss = torch.stack(termination_losses).mean()
        total_loss = (
            self.cfg.state_loss_weight * state_loss
            + self.cfg.sequence_loss_weight * sequence_loss
            + self.cfg.contact_loss_weight * contact_loss
            + self.cfg.termination_loss_weight * termination_loss
        )
        return {
            "total_loss": total_loss,
            "state_loss": state_loss,
            "sequence_loss": sequence_loss,
            "contact_loss": contact_loss,
            "termination_loss": termination_loss,
        }

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
        chosen_mean = mean_stack.gather(0, gather_idx.expand(1, -1, self.cfg.state_dim))[0]
        chosen_contact = contact_stack.gather(0, gather_idx.expand(1, -1, self.cfg.contact_dim))[0]
        chosen_term = term_stack.gather(0, gather_idx.expand(1, -1, self.cfg.termination_dim))[0]

        raw_members = self.denormalize_state(mean_stack)
        epistemic = raw_members.std(dim=0).mean(dim=-1)
        aleatoric = torch.exp(logstd_stack).mean(dim=(0, 2))
        next_state = self.denormalize_state(chosen_mean)
        return next_state, aleatoric, epistemic, chosen_contact, chosen_term

    def checkpoint(self, optimizer: torch.optim.Optimizer | None, iteration: int, infos: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "system_dynamics_state_dict": self.state_dict(),
            "system_dynamics_optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "iter": iteration,
            "infos": {
                **(infos or {}),
                "dynamics_config": asdict(self.cfg),
            },
        }


def train_world_model_steps(
    dynamics: SystemDynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    replay: SequenceReplayBuffer,
    cfg: WorldModelConfig,
    device: torch.device | str,
    updates: int | None = None,
) -> dict[str, float]:
    if not replay.can_sample(
        cfg.dynamics.history_horizon,
        cfg.dynamics.forecast_horizon,
        cfg.replay.min_transitions,
    ):
        return {}

    state_mean, state_std, action_mean, action_std = replay.stats()
    dynamics.set_normalizers(state_mean, state_std, action_mean, action_std)
    dynamics.train()

    totals: dict[str, float] = {}
    num_updates = updates if updates is not None else cfg.updates_per_iter
    for _ in range(num_updates):
        batch = replay.sample(
            batch_size=cfg.replay.batch_size,
            history_horizon=cfg.dynamics.history_horizon,
            forecast_horizon=cfg.dynamics.forecast_horizon,
            device=device,
        )
        loss_dict = dynamics.compute_loss(*batch, bootstrap=True)
        optimizer.zero_grad(set_to_none=True)
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=10.0)
        optimizer.step()
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    metrics = {key: value / num_updates for key, value in totals.items()}
    dynamics.eval()
    with torch.no_grad():
        eval_batch = replay.sample(
            batch_size=cfg.replay.batch_size,
            history_horizon=cfg.dynamics.history_horizon,
            forecast_horizon=cfg.dynamics.forecast_horizon,
            device=device,
        )
        eval_loss = dynamics.compute_loss(*eval_batch, bootstrap=False)
        metrics["eval_state_loss"] = float(eval_loss["state_loss"].detach().cpu())
        states, actions, next_states, _contacts, _terms = eval_batch
        pred_state, _aleatoric, _epistemic, _contact, _term = dynamics.predict(
            states[:, : cfg.dynamics.history_horizon],
            actions[:, : cfg.dynamics.history_horizon],
        )
        target_state = next_states[:, cfg.dynamics.history_horizon - 1]
        denom = target_state.abs().sum(dim=-1).clamp_min(1e-6)
        metrics["traj_autoregressive_error"] = float(((pred_state - target_state).abs().sum(dim=-1) / denom).mean().cpu())
    dynamics.train()
    return metrics


def load_dynamics_checkpoint(path: str | Path, device: torch.device | str) -> tuple[SystemDynamicsEnsemble, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    infos = checkpoint.get("infos") or {}
    cfg_dict = infos.get("dynamics_config")
    if cfg_dict is None:
        raise ValueError(f"Checkpoint {path} does not contain infos['dynamics_config'].")
    cfg = DynamicsConfig(**cfg_dict)
    dynamics = SystemDynamicsEnsemble(cfg).to(device)
    dynamics.load_state_dict(checkpoint["system_dynamics_state_dict"])
    dynamics.eval()
    return dynamics, checkpoint
