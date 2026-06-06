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
            "format_version": "go2_mixed_rwm_dataset_v1",
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "contact_dim": int(contact_dim),
            "termination_dim": int(termination_dim),
            "num_envs": int(num_envs),
            "capacity": int(capacity),
            "states": [],
            "actions": [],
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
    ) -> None:
        self.data["observations"].append(_detach_cpu(obs, torch.float32))
        self.data["next_observations"].append(_detach_cpu(next_obs, torch.float32))
        self.data["states"].append(_detach_cpu(state, torch.float32))
        self.data["actions"].append(_detach_cpu(action, torch.float32))
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
        self.state_dim = int(self.states.shape[-1])
        self.action_dim = int(self.actions.shape[-1])
        self.contact_dim = int(self.contacts.shape[-1])
        self.termination_dim = int(self.terminations.shape[-1])
        self.seq_len = int(cfg.history_horizon + cfg.forecast_horizon)
        if self.num_time_steps < self.seq_len:
            raise ValueError(
                f"Dataset has only {self.num_time_steps} time steps; need at least {self.seq_len}."
            )
        self.train_indices, self.val_indices = self._build_indices()

    @property
    def num_transitions(self) -> int:
        return self.num_time_steps * self.num_envs

    def _build_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.num_time_steps - self.seq_len
        starts: list[int] = []
        env_ids: list[int] = []
        term_bool = self.terminations.squeeze(-1).bool()
        for start in range(max_start + 1):
            window_has_boundary = term_bool[start : start + self.seq_len - 1].any(dim=0)
            valid_envs = (~window_has_boundary).nonzero(as_tuple=False).flatten()
            starts.extend([start] * int(valid_envs.numel()))
            env_ids.extend(valid_envs.tolist())
        if not starts:
            starts = [int(torch.randint(0, max_start + 1, ()).item())]
            env_ids = [int(torch.randint(0, self.num_envs, ()).item())]
        all_indices = torch.stack(
            [torch.tensor(starts, dtype=torch.long), torch.tensor(env_ids, dtype=torch.long)],
            dim=1,
        )
        generator = torch.Generator().manual_seed(int(self.cfg.seed))
        perm = torch.randperm(all_indices.shape[0], generator=generator)
        all_indices = all_indices[perm]
        split = int(math.floor(all_indices.shape[0] * float(self.cfg.train_fraction)))
        split = max(1, min(split, all_indices.shape[0] - 1)) if all_indices.shape[0] > 1 else 1
        return all_indices[:split], all_indices[split:] if split < all_indices.shape[0] else all_indices[:1]

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.train_indices if split == "train" else self.val_indices
        if indices.numel() == 0:
            indices = self.train_indices
        seq_len = self.cfg.history_horizon + int(forecast_horizon or self.cfg.forecast_horizon)
        max_start = self.num_time_steps - seq_len
        sample_ids = torch.randint(0, indices.shape[0], (batch_size,))
        chosen = indices[sample_ids]
        out_states = []
        out_actions = []
        out_next_states = []
        out_contacts = []
        out_terms = []
        for start, env_id in chosen.tolist():
            start = min(int(start), max_start)
            sl = slice(start, start + seq_len)
            out_states.append(self.states[sl, env_id])
            out_actions.append(self.actions[sl, env_id])
            out_next_states.append(self.next_states[sl, env_id])
            out_contacts.append(self.contacts[sl, env_id])
            out_terms.append(self.terminations[sl, env_id])
        return (
            torch.stack(out_states, dim=0).to(device),
            torch.stack(out_actions, dim=0).to(device),
            torch.stack(out_next_states, dim=0).to(device),
            torch.stack(out_contacts, dim=0).to(device),
            torch.stack(out_terms, dim=0).to(device),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "num_transitions": self.num_transitions,
            "num_time_steps": self.num_time_steps,
            "num_envs": self.num_envs,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "contact_dim": self.contact_dim,
            "termination_dim": self.termination_dim,
            "train_sequences": int(self.train_indices.shape[0]),
            "val_sequences": int(self.val_indices.shape[0]),
            "source_metadata": self.dataset.get("metadata") or {},
        }
