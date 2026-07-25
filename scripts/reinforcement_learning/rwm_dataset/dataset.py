"""Dataset helpers for Go2 mixed offline RWM training.

The saved dictionary intentionally contains the legacy keys used by
``SequenceReplayBuffer`` so existing RWM imagination scripts can load it
without changes. Extra keys store richer transition data for offline analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


COLLECTOR_NAME_TO_ID = {
    "random": 0,
    "expert": 1,
    "noisy_expert": 2,
    "medium": 3,
    "failure_border": 4,
}
COLLECTOR_ID_TO_NAME = {v: k for k, v in COLLECTOR_NAME_TO_ID.items()}


def parse_collector_mix(spec: str) -> dict[str, float]:
    """Parse strings such as ``random:0.15,expert:0.85``."""

    result: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", maxsplit=1)
        name = name.strip()
        if name not in COLLECTOR_NAME_TO_ID:
            allowed = ", ".join(COLLECTOR_NAME_TO_ID)
            raise ValueError(f"Unknown collector type {name!r}. Allowed: {allowed}")
        result[name] = float(value)
    if not result:
        raise ValueError("collector_mix must contain at least one collector.")
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("collector_mix weights must sum to a positive value.")
    return {key: value / total for key, value in result.items()}


def sample_collector_ids(mix: dict[str, float], num_envs: int, device: torch.device | str) -> torch.Tensor:
    names = list(mix)
    probs = torch.tensor([mix[name] for name in names], device=device, dtype=torch.float32)
    ids = torch.tensor([COLLECTOR_NAME_TO_ID[name] for name in names], device=device, dtype=torch.long)
    sampled = torch.multinomial(probs, num_envs, replacement=True)
    return ids[sampled]


def _detach_cpu(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    out = value.detach().cpu()
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _detach_cpu_tree(value: Any) -> Any:
    """Copy a nested simulator snapshot to CPU without changing its schema."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _detach_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cpu_tree(item) for item in value)
    return value


SIMULATOR_SNAPSHOT_DATASET_KEYS = {
    "root_state_local": "sim_root_states_local",
    "joint_position": "sim_joint_positions",
    "joint_velocity": "sim_joint_velocities",
    "action": "sim_action_histories",
    "prev_action": "sim_prev_action_histories",
    "prev_prev_action": "sim_prev_prev_action_histories",
    "command": "sim_snapshot_commands",
}


class Go2MixedDatasetBuilder:
    """Accumulates vectorized transitions in legacy-compatible list form."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        contact_dim: int,
        termination_dim: int,
        num_envs: int,
        capacity: int,
        metadata: dict[str, Any],
    ) -> None:
        self.data: dict[str, Any] = {
            "format_version": "go2_mixed_rwm_dataset_v2",
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "contact_dim": int(contact_dim),
            "termination_dim": int(termination_dim),
            "num_envs": int(num_envs),
            "capacity": int(capacity),
            "states": [],
            "actions": [],
            "raw_actions": [],
            "env_action_delay_steps": [],
            "actuator_delay_substeps": [],
            "next_states": [],
            "contacts": [],
            "terminations": [],
            "observations": [],
            "next_observations": [],
            "commands": [],
            "rewards": [],
            "dones": [],
            "timeouts": [],
            "prev_actions": [],
            "episode_ids": [],
            "timesteps": [],
            "collector_types": [],
            "noisy_actor_observations": [],
            "metadata": dict(metadata),
        }

    @property
    def num_time_steps(self) -> int:
        return len(self.data["states"])

    @property
    def num_transitions(self) -> int:
        return self.num_time_steps * int(self.data["num_envs"])

    def add(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        contact: torch.Tensor,
        termination: torch.Tensor,
        command: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        timeout: torch.Tensor,
        prev_action: torch.Tensor,
        episode_id: torch.Tensor,
        timestep: torch.Tensor,
        collector_type: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        env_action_delay_step: torch.Tensor | None = None,
        actuator_delay_substep: torch.Tensor | None = None,
        noisy_actor_observation: torch.Tensor | None = None,
        simulator_snapshot: dict[str, Any] | None = None,
        trace_reset_reconstruction_error: torch.Tensor | None = None,
        trace_valid_mask: torch.Tensor | None = None,
        trace_foot_site_position_w: torch.Tensor | None = None,
        trace_foot_site_linear_velocity_w: torch.Tensor | None = None,
    ) -> None:
        self.data["observations"].append(_detach_cpu(obs, torch.float32))
        self.data["next_observations"].append(_detach_cpu(next_obs, torch.float32))
        self.data["states"].append(_detach_cpu(state, torch.float32))
        self.data["actions"].append(_detach_cpu(action, torch.float32))
        self.data["raw_actions"].append(
            _detach_cpu(action if raw_action is None else raw_action, torch.float32)
        )
        num_envs = int(action.shape[0])
        if env_action_delay_step is None:
            env_action_delay_step = torch.zeros(num_envs, device=action.device, dtype=torch.long)
        if actuator_delay_substep is None:
            actuator_delay_substep = torch.zeros(num_envs, device=action.device, dtype=torch.long)
        self.data["env_action_delay_steps"].append(_detach_cpu(env_action_delay_step, torch.long))
        self.data["actuator_delay_substeps"].append(_detach_cpu(actuator_delay_substep, torch.long))
        self.data["next_states"].append(_detach_cpu(next_state, torch.float32))
        self.data["contacts"].append(_detach_cpu(contact, torch.float32))
        self.data["terminations"].append(_detach_cpu(termination, torch.float32))
        self.data["commands"].append(_detach_cpu(command, torch.float32))
        self.data["rewards"].append(_detach_cpu(reward, torch.float32))
        self.data["dones"].append(_detach_cpu(done, torch.bool))
        self.data["timeouts"].append(_detach_cpu(timeout, torch.bool))
        self.data["prev_actions"].append(_detach_cpu(prev_action, torch.float32))
        self.data["episode_ids"].append(_detach_cpu(episode_id, torch.long))
        self.data["timesteps"].append(_detach_cpu(timestep, torch.long))
        self.data["collector_types"].append(_detach_cpu(collector_type, torch.long))
        if noisy_actor_observation is None:
            noisy_actor_observation = obs
        self.data["noisy_actor_observations"].append(
            _detach_cpu(noisy_actor_observation, torch.float32)
        )
        if simulator_snapshot is not None:
            missing = [
                key for key in SIMULATOR_SNAPSHOT_DATASET_KEYS
                if key not in simulator_snapshot
            ]
            if missing:
                raise ValueError(f"Simulator snapshot is missing fields: {missing}")
            for snapshot_key, dataset_key in SIMULATOR_SNAPSHOT_DATASET_KEYS.items():
                self.data.setdefault(dataset_key, []).append(
                    _detach_cpu_tree(simulator_snapshot[snapshot_key])
                )
            self.data["metadata"]["simulator_snapshot_version"] = simulator_snapshot.get(
                "snapshot_version"
            )
        if trace_reset_reconstruction_error is not None:
            self.data.setdefault("trace_reset_reconstruction_errors", []).append(
                _detach_cpu(trace_reset_reconstruction_error, torch.float32)
            )
        if trace_valid_mask is not None:
            self.data.setdefault("trace_valid_masks", []).append(
                _detach_cpu(trace_valid_mask, torch.bool)
            )
        if trace_foot_site_position_w is not None:
            self.data.setdefault("trace_foot_site_positions_w", []).append(
                _detach_cpu(trace_foot_site_position_w, torch.float32)
            )
        if trace_foot_site_linear_velocity_w is not None:
            self.data.setdefault("trace_foot_site_linear_velocities_w", []).append(
                _detach_cpu(trace_foot_site_linear_velocity_w, torch.float32)
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data["metadata"]["num_time_steps"] = self.num_time_steps
        self.data["metadata"]["num_transitions"] = self.num_transitions
        torch.save(self.data, path)


def load_mixed_dataset(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def merge_dataset_dicts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("No dataset parts to merge.")
    merged = dict(parts[0])
    list_keys = [key for key, value in merged.items() if isinstance(value, list)]
    for key in list_keys:
        merged[key] = []
    for part in parts:
        for key in list_keys:
            merged[key].extend(part.get(key, []))
    metadata = dict(parts[0].get("metadata") or {})
    metadata["merged_parts"] = len(parts)
    metadata["num_time_steps"] = len(merged["states"])
    metadata["num_transitions"] = len(merged["states"]) * int(merged["num_envs"])
    merged["metadata"] = metadata
    merged["capacity"] = metadata["num_transitions"]
    return merged


def save_dataset_dict(dataset: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)


def stack_time_key(dataset: dict[str, Any], key: str) -> torch.Tensor:
    values = dataset.get(key)
    if values is None:
        raise KeyError(f"Dataset is missing key {key!r}.")
    if isinstance(values, torch.Tensor):
        return values
    if not isinstance(values, list) or not values:
        raise ValueError(f"Dataset key {key!r} must be a non-empty list or tensor.")
    return torch.stack(values, dim=0)


@dataclass
class OfflineSamplerConfig:
    history_horizon: int = 32
    forecast_horizon: int = 8
    train_fraction: float = 0.95
    seed: int = 0
    terminal_window_fraction: float = 0.0


class OfflineSequenceSampler:
    """Samples contiguous windows from a saved vectorized dataset."""

    def __init__(self, dataset: dict[str, Any], cfg: OfflineSamplerConfig) -> None:
        self.dataset = dataset
        self.cfg = cfg
        self.states = stack_time_key(dataset, "states").float()
        self.actions = stack_time_key(dataset, "actions").float()
        self.next_states = stack_time_key(dataset, "next_states").float()
        self.contacts = stack_time_key(dataset, "contacts").float()
        self.terminations = stack_time_key(dataset, "terminations").float()
        self.num_time_steps = int(self.states.shape[0])
        self.num_envs = int(self.states.shape[1])
        confidence_values = dataset.get("next_base_lin_vel_confidence")
        if confidence_values is None:
            self.next_base_lin_vel_confidence = torch.ones(
                self.num_time_steps,
                self.num_envs,
                dtype=torch.float32,
            )
            self.has_base_lin_vel_confidence = False
        else:
            self.next_base_lin_vel_confidence = stack_time_key(
                dataset, "next_base_lin_vel_confidence"
            ).float()
            if (
                self.next_base_lin_vel_confidence.ndim == 3
                and self.next_base_lin_vel_confidence.shape[-1] == 1
            ):
                self.next_base_lin_vel_confidence = self.next_base_lin_vel_confidence.squeeze(-1)
            expected_confidence_shape = (self.num_time_steps, self.num_envs)
            if tuple(self.next_base_lin_vel_confidence.shape) != expected_confidence_shape:
                raise ValueError(
                    "next_base_lin_vel_confidence shape is "
                    f"{tuple(self.next_base_lin_vel_confidence.shape)}, "
                    f"expected {expected_confidence_shape}"
                )
            if not torch.all(
                (self.next_base_lin_vel_confidence >= 0.0)
                & (self.next_base_lin_vel_confidence <= 1.0)
            ):
                raise ValueError("next_base_lin_vel_confidence must be in [0, 1]")
            self.has_base_lin_vel_confidence = True
        episode_values = dataset.get("episode_ids")
        self.episode_ids = (
            stack_time_key(dataset, "episode_ids").long()
            if episode_values is not None
            else None
        )
        self.state_dim = int(self.states.shape[-1])
        self.action_dim = int(self.actions.shape[-1])
        self.contact_dim = int(self.contacts.shape[-1])
        self.termination_dim = int(self.terminations.shape[-1])
        self.seq_len = int(cfg.history_horizon + cfg.forecast_horizon)
        if not 0.0 <= float(cfg.terminal_window_fraction) < 1.0:
            raise ValueError("terminal_window_fraction must be in [0, 1)")
        if self.num_time_steps < self.seq_len:
            raise ValueError(
                f"Dataset has only {self.num_time_steps} time steps; need at least {self.seq_len}."
            )
        self.train_indices, self.val_indices = self._build_indices()
        self.train_terminal_indices = self._terminal_indices(self.train_indices)
        self.val_terminal_indices = self._terminal_indices(self.val_indices)

    @property
    def num_transitions(self) -> int:
        return self.num_time_steps * self.num_envs

    def _build_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The model consumes transitions through seq_len - 2; the final row is
        # only a shape-compatible lookahead slot and may be padded at EOF.
        max_start = self.num_time_steps - (self.seq_len - 1)
        starts: list[int] = []
        env_ids: list[int] = []
        episode_ids: list[int] = []
        term_bool = self.terminations.reshape(
            self.num_time_steps,
            self.num_envs,
            -1,
        ).bool().any(dim=-1)
        for start in range(max_start + 1):
            # The dynamics loss consumes transitions through relative index
            # seq_len - 2.  A terminal at that final target is valid; only a
            # terminal before it would make the forecast cross an episode.
            window_has_boundary = term_bool[start : start + self.seq_len - 2].any(dim=0)
            if self.episode_ids is not None:
                episode_change = (
                    self.episode_ids[start : start + self.seq_len - 2]
                    != self.episode_ids[start + 1 : start + self.seq_len - 1]
                ).any(dim=0)
                window_has_boundary |= episode_change
            valid_envs = (~window_has_boundary).nonzero(as_tuple=False).flatten()
            starts.extend([start] * int(valid_envs.numel()))
            env_ids.extend(valid_envs.tolist())
            if self.episode_ids is not None:
                episode_ids.extend(self.episode_ids[start, valid_envs].tolist())
        if not starts:
            starts = [int(torch.randint(0, max_start + 1, ()).item())]
            env_ids = [int(torch.randint(0, self.num_envs, ()).item())]
            if self.episode_ids is not None:
                episode_ids = [int(self.episode_ids[starts[0], env_ids[0]].item())]
        all_indices = torch.stack(
            [torch.tensor(starts, dtype=torch.long), torch.tensor(env_ids, dtype=torch.long)],
            dim=1,
        )
        generator = torch.Generator().manual_seed(int(self.cfg.seed))
        if self.episode_ids is not None and len(set(episode_ids)) > 1:
            window_episode_ids = torch.tensor(episode_ids, dtype=torch.long)
            unique_episode_ids = torch.unique(window_episode_ids)
            episode_perm = torch.randperm(unique_episode_ids.numel(), generator=generator)
            episode_split = int(math.floor(unique_episode_ids.numel() * float(self.cfg.train_fraction)))
            episode_split = max(1, min(episode_split, unique_episode_ids.numel() - 1))
            train_episode_ids = unique_episode_ids[episode_perm[:episode_split]]
            train_mask = torch.isin(window_episode_ids, train_episode_ids)
            train_indices = all_indices[train_mask]
            val_indices = all_indices[~train_mask]
            train_indices = train_indices[torch.randperm(train_indices.shape[0], generator=generator)]
            val_indices = val_indices[torch.randperm(val_indices.shape[0], generator=generator)]
            return train_indices, val_indices

        perm = torch.randperm(all_indices.shape[0], generator=generator)
        all_indices = all_indices[perm]
        split = int(math.floor(all_indices.shape[0] * float(self.cfg.train_fraction)))
        split = max(1, min(split, all_indices.shape[0] - 1)) if all_indices.shape[0] > 1 else 1
        return all_indices[:split], all_indices[split:] if split < all_indices.shape[0] else all_indices[:1]

    def _terminal_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return indices
        final_target_offset = self.seq_len - 2
        target_times = indices[:, 0] + final_target_offset
        target_envs = indices[:, 1]
        terminal = self.terminations[target_times, target_envs].reshape(
            indices.shape[0], -1
        ).bool().any(dim=-1)
        return indices[terminal]

    def stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.cat(
            [
                self.states.reshape(-1, self.state_dim),
                self.next_states.reshape(-1, self.state_dim),
            ],
            dim=0,
        )
        actions = self.actions.reshape(-1, self.action_dim)
        return (
            states.mean(dim=0),
            states.std(dim=0).clamp_min(1e-6),
            actions.mean(dim=0),
            actions.std(dim=0).clamp_min(1e-6),
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        split: str = "train",
        forecast_horizon: int | None = None,
        include_base_lin_vel_confidence: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        indices = self.train_indices if split == "train" else self.val_indices
        if indices.numel() == 0:
            indices = self.train_indices
        seq_len = self.cfg.history_horizon + int(forecast_horizon or self.cfg.forecast_horizon)
        max_start = self.num_time_steps - (seq_len - 1)
        terminal_indices = (
            self.train_terminal_indices if split == "train" else self.val_terminal_indices
        )
        terminal_count = 0
        if split == "train" and terminal_indices.numel() > 0:
            terminal_count = min(
                batch_size,
                int(round(batch_size * float(self.cfg.terminal_window_fraction))),
            )
        ordinary_count = batch_size - terminal_count
        chosen_parts = []
        if ordinary_count:
            sample_ids = torch.randint(0, indices.shape[0], (ordinary_count,))
            chosen_parts.append(indices[sample_ids])
        if terminal_count:
            terminal_sample_ids = torch.randint(
                0, terminal_indices.shape[0], (terminal_count,)
            )
            chosen_parts.append(terminal_indices[terminal_sample_ids])
        chosen = torch.cat(chosen_parts, dim=0)
        if chosen.shape[0] > 1:
            chosen = chosen[torch.randperm(chosen.shape[0])]
        out_states = []
        out_actions = []
        out_next_states = []
        out_contacts = []
        out_terms = []
        out_base_lin_vel_confidence = []
        for start, env_id in chosen.tolist():
            start = min(int(start), max_start)
            sl = slice(start, start + seq_len)
            def padded_window(value: torch.Tensor) -> torch.Tensor:
                window = value[sl, env_id]
                if window.shape[0] == seq_len - 1:
                    window = torch.cat([window, window[-1:].clone()], dim=0)
                if window.shape[0] != seq_len:
                    raise RuntimeError(
                        f"sampled window has {window.shape[0]} rows, expected {seq_len}"
                    )
                return window

            out_states.append(padded_window(self.states))
            out_actions.append(padded_window(self.actions))
            out_next_states.append(padded_window(self.next_states))
            out_contacts.append(padded_window(self.contacts))
            out_terms.append(padded_window(self.terminations))
            out_base_lin_vel_confidence.append(
                padded_window(self.next_base_lin_vel_confidence)
            )
        batch = (
            torch.stack(out_states, dim=0).to(device),
            torch.stack(out_actions, dim=0).to(device),
            torch.stack(out_next_states, dim=0).to(device),
            torch.stack(out_contacts, dim=0).to(device),
            torch.stack(out_terms, dim=0).to(device),
        )
        if include_base_lin_vel_confidence:
            return (*batch, torch.stack(out_base_lin_vel_confidence, dim=0).to(device))
        return batch

    def metadata(self) -> dict[str, Any]:
        return {
            "num_transitions": self.num_transitions,
            "num_time_steps": self.num_time_steps,
            "num_envs": self.num_envs,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "contact_dim": self.contact_dim,
            "termination_dim": self.termination_dim,
            "has_base_lin_vel_confidence": self.has_base_lin_vel_confidence,
            "train_sequences": int(self.train_indices.shape[0]),
            "val_sequences": int(self.val_indices.shape[0]),
            "train_terminal_sequences": int(self.train_terminal_indices.shape[0]),
            "val_terminal_sequences": int(self.val_terminal_indices.shape[0]),
            "terminal_window_fraction": float(self.cfg.terminal_window_fraction),
            "source_metadata": self.dataset.get("metadata") or {},
        }
