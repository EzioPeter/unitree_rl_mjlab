"""Current-actor proposal generation from V1 buffer snapshots."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .simulator_reset import (
    SNAPSHOT_REQUIRED_KEYS,
    SNAPSHOT_VERSION,
    _force_commands,
    repeat_snapshot_rows,
    reset_from_snapshot,
    step_without_automatic_reset,
    validate_snapshot,
)


def snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    validate_snapshot(snapshot)
    digest = hashlib.sha256(SNAPSHOT_VERSION.encode("utf-8"))
    for key in SNAPSHOT_REQUIRED_KEYS:
        value = torch.as_tensor(snapshot[key]).detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProposalConfig:
    rollout_horizon: int
    trajectories_per_start: int
    actor_sample_temperature: float = 1.0
    candidate_start_mode: str = "buffer_snapshot"
    random_reset_fraction: float = 0.0

    def validate(self) -> None:
        if self.rollout_horizon < 1 or self.trajectories_per_start < 2:
            raise ValueError("Proposal horizon must be positive and branches at least two.")
        if (
            not math.isfinite(float(self.actor_sample_temperature))
            or self.actor_sample_temperature <= 0.0
        ):
            raise ValueError("Go2 TRACE proposal actor temperature must be finite and positive.")
        if self.candidate_start_mode != "buffer_snapshot":
            raise ValueError("Formal proposals start only from buffer_snapshot.")
        if self.random_reset_fraction != 0.0:
            raise ValueError("Native random reset is not a formal TRACE candidate source.")


class FlashSACActorDistributionSampler:
    """Sample only from the current FlashSAC actor distribution."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor

    def _mean_std(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.actor, "apply"):
            result = self.actor.apply(
                "get_mean_and_std",
                observations=observations,
                training=False,
            )
        elif hasattr(self.actor, "get_mean_and_std"):
            result = self.actor.get_mean_and_std(
                observations=observations,
                training=False,
            )
        else:
            raise TypeError("FlashSAC actor lacks get_mean_and_std.")
        mean, std = result
        if mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("FlashSAC actor returned malformed distribution parameters.")
        if bool((std <= 0.0).any()):
            raise ValueError("FlashSAC actor returned non-positive standard deviation.")
        return mean, std

    def sample(
        self,
        observations: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float,
    ) -> torch.Tensor:
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("Proposal actor temperature must be finite and positive.")
        # Proposal generation is actor inference only. Without no_grad, every
        # simulator step retains the full 4096-row actor autograd graph through
        # MJLab's action buffers, adding roughly 1 GiB per refresh.
        with torch.no_grad():
            mean, std = self._mean_std(observations)
            # CPU RNG isolates proposal randomness from training/CUDA global RNG.
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device="cpu",
                generator=generator,
            ).to(mean.device)
            return torch.tanh(mean + float(temperature) * std * noise)


class V13MJLabProposalCollector:
    """Collect branches in an already configured V13 imperfect simulator.

    Environment construction remains V13-owned. This adapter only performs
    V1 restoration, current-actor sampling, and no-auto-reset rollout.
    """

    def __init__(
        self,
        *,
        env: Any,
        extractor: Any,
        actor_sampler: FlashSACActorDistributionSampler,
        make_policy_observation: Any,
        simulator_config_sha256: str,
        randomization_disabled: bool,
        policy_observation_mask_indices: Sequence[int] = (),
    ) -> None:
        if not simulator_config_sha256:
            raise ValueError("Proposal collector requires simulator config provenance.")
        if randomization_disabled is not True:
            raise ValueError(
                "Formal TRACE proposals do not add gravity/friction/joint-noise/DR mismatch."
            )
        self.env = getattr(env, "unwrapped", env)
        self.extractor = extractor
        self.actor_sampler = actor_sampler
        self.make_policy_observation = make_policy_observation
        self.simulator_config_sha256 = str(simulator_config_sha256)
        self.policy_observation_mask_indices = tuple(
            sorted(set(map(int, policy_observation_mask_indices)))
        )

    def _actor_observation(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        actor = self.make_policy_observation(state, command, previous_action)
        if actor.ndim != 2 or actor.shape[1] != 48:
            raise ValueError("V13 make_go2_policy_obs must return [N,48].")
        if self.policy_observation_mask_indices:
            mask = set(self.policy_observation_mask_indices)
            keep = [index for index in range(actor.shape[1]) if index not in mask]
            actor = actor[:, keep]
        return actor

    def _root_state_local(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Capture only the root field consumed by the trajectory summary.

        ``capture_snapshot_v1`` also clones joints, three action histories, and
        commands.  Proposal collection discarded all of those fields on every
        rollout step, so keeping this narrow capture avoids redundant device
        work without changing the recorded root-state values.
        """

        robot = self.env.scene["robot"]
        root = torch.cat(
            [robot.data.root_link_pose_w, robot.data.root_link_vel_w],
            dim=-1,
        )[env_ids].clone()
        root[:, :3] -= self.env.scene.env_origins[env_ids]
        return root

    def collect(
        self,
        snapshot: Mapping[str, Any],
        *,
        start_state_ids: Sequence[str],
        config: ProposalConfig,
        actor_generator: torch.Generator,
        proposal_event: int,
    ) -> list[dict[str, Any]]:
        config.validate()
        validate_snapshot(snapshot, len(start_state_ids))
        expanded = repeat_snapshot_rows(snapshot, config.trajectories_per_start)
        num_envs = len(start_state_ids) * config.trajectories_per_start
        if int(self.env.num_envs) != num_envs:
            raise ValueError(
                f"Proposal env has {self.env.num_envs} worlds but event needs {num_envs}."
            )
        env_ids = torch.arange(num_envs, device=self.env.device, dtype=torch.long)
        reset_from_snapshot(self.env, expanded, env_ids)
        commands = torch.as_tensor(
            expanded["command"], device=self.env.device, dtype=torch.float32
        )
        previous_action = torch.as_tensor(
            expanded["action"], device=self.env.device, dtype=torch.float32
        )
        source_hashes = []
        for source_index in range(len(start_state_ids)):
            one = {
                "snapshot_version": SNAPSHOT_VERSION,
                **{
                    key: torch.as_tensor(snapshot[key])[source_index : source_index + 1]
                    for key in SNAPSHOT_REQUIRED_KEYS
                },
            }
            source_hashes.append(snapshot_sha256(one))
        field_names = (
            "states",
            "actions",
            "next_states",
            "contacts",
            "terminations",
            "truncations",
            "commands",
            "rewards",
            "prev_actions",
            "sim_root_states_local",
        )
        # Retain one batched tensor per rollout step.  The historical path
        # performed one synchronous GPU->CPU copy for every live environment
        # and every field (roughly 700k tiny copies per formal proposal).
        # Stacking [env, time, ...] first reduces that to one bulk copy per
        # field while preserving the same per-trajectory ordering.
        batched_steps: dict[str, list[torch.Tensor]] = {
            key: [] for key in field_names
        }
        alive_steps: list[torch.Tensor] = []
        alive = torch.ones(num_envs, dtype=torch.bool, device=self.env.device)

        for _step in range(config.rollout_horizon):
            if not bool(alive.any()):
                break
            alive_steps.append(alive.detach().clone())
            _force_commands(self.env, commands, env_ids)
            state = self.extractor.extract_state().detach().clone()
            actor_observation = self._actor_observation(
                state, commands, previous_action
            )
            action = self.actor_sampler.sample(
                actor_observation,
                generator=actor_generator,
                temperature=config.actor_sample_temperature,
            )
            _, reward, terminated, truncated, _ = step_without_automatic_reset(
                self.env, action
            )
            next_state = self.extractor.extract_state().detach().clone()
            contacts = self.extractor.extract_contact().detach().clone()
            root_state_local = self._root_state_local(env_ids)
            done = terminated.bool() | truncated.bool()
            step_values = {
                "states": state,
                "actions": action,
                "next_states": next_state,
                "contacts": contacts,
                "terminations": terminated.float(),
                "truncations": truncated.float(),
                "commands": commands,
                "rewards": reward,
                "prev_actions": previous_action,
                "sim_root_states_local": root_state_local,
            }
            for key, value in step_values.items():
                batched_steps[key].append(value.detach().clone())
            alive &= ~done
            previous_action = action.detach().clone()

        if not alive_steps:
            raise RuntimeError("A proposal event produced no rollout step.")
        alive_by_env = torch.stack(alive_steps, dim=1).detach().cpu()
        device_values_by_env = {
            key: torch.stack(values, dim=1).detach()
            for key, values in batched_steps.items()
        }
        values_by_env = {
            key: value.cpu()
            for key, value in device_values_by_env.items()
        }
        del batched_steps, alive_steps
        trajectories: list[dict[str, Any]] = []
        for env_index in range(num_envs):
            length = int(alive_by_env[env_index].sum().item())
            if length < 1:
                raise RuntimeError("A proposal branch produced zero valid transitions.")
            source_index = env_index // config.trajectories_per_start
            branch_index = env_index % config.trajectories_per_start
            trajectory = {
                key: value[env_index, :length]
                for key, value in values_by_env.items()
            }
            # Summaries use the CPU tensors above.  Selected trajectories can
            # reuse these device views for reward materialization, avoiding a
            # CPU->GPU round trip for the selected subset.
            trajectory["_device_fields"] = {
                key: value[env_index, :length]
                for key, value in device_values_by_env.items()
            }
            identity = (
                f"event{proposal_event:08d}_start{start_state_ids[source_index]}"
                f"_branch{branch_index:03d}"
            )
            trajectory.update(
                {
                    "trajectory_id": identity,
                    "start_state_id": str(start_state_ids[source_index]),
                    "start_state_key": source_hashes[source_index],
                    "comparison_group_key": source_hashes[source_index],
                    "realized_snapshot_hash": source_hashes[source_index],
                    "simulator_config_hash": self.simulator_config_sha256,
                    "source_kind": "same_start_candidate",
                    "candidate_seed": int(proposal_event),
                    "step_dt": float(self.env.step_dt),
                    "expected_trajectory_length": config.rollout_horizon,
                }
            )
            trajectories.append(trajectory)
        return trajectories
