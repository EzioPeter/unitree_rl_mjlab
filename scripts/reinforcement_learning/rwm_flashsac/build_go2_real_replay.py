"""Build a reward-aligned n-step real replay artifact for V12 FlashSAC.

The source dataset stores 45D policy observations without base linear
velocity.  This builder reconstructs the 48D critic observations, recomputes
the public RWM reward on real transitions, and creates episode-safe n-step
transitions.  It deliberately does not use the source dataset's task reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from scripts.reinforcement_learning.rwm_flashsac.world_model_env import (
    FlashSACWorldModelEnvConfig,
    configure_go2_reward_state,
)
from src.tasks.rwm_velocity.mdp.rewards import (
    Go2RWMRewardState,
    compute_go2_imagination_reward,
)


FORMAT_VERSION = "go2_real_replay_v1"
REPLAY_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dataset-size", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _stack_time_key(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
        tensor = torch.stack(value)
    else:
        raise ValueError(f"Dataset key {key!r} must be a non-empty tensor or tensor list.")
    if tensor.ndim < 2:
        raise ValueError(f"Dataset key {key!r} must start with [time, env], got {tuple(tensor.shape)}.")
    return tensor.detach().cpu()


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match rank {len(expected)}.")
    for actual, wanted in zip(tensor.shape, expected, strict=True):
        if wanted is not None and int(actual) != wanted:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match {expected}.")


def _configure_reward_state(
    cfg: FlashSACWorldModelEnvConfig,
    *,
    num_envs: int,
    action_dim: int,
    device: torch.device,
) -> Go2RWMRewardState:
    """Mirror the reward-state configuration used by world_model_env.py."""

    return configure_go2_reward_state(
        cfg,
        num_envs=num_envs,
        action_dim=action_dim,
        device=device,
    )


def _validate_dataset_layout(tensors: dict[str, torch.Tensor]) -> tuple[int, int]:
    states = tensors["states"].float()
    actions = tensors["actions"].float()
    observations = tensors["observations"].float()
    next_states = tensors["next_states"].float()
    next_observations = tensors["next_observations"].float()
    commands = tensors["commands"].float()
    contacts = tensors["contacts"].float()
    prev_actions = tensors["prev_actions"].float()
    episode_ids = tensors["episode_ids"].long()
    timesteps = tensors["timesteps"].long()

    time_steps, num_envs = map(int, states.shape[:2])
    prefix = (time_steps, num_envs)
    _require_shape("states", states, (*prefix, 45))
    _require_shape("next_states", next_states, (*prefix, 45))
    _require_shape("observations", observations, (*prefix, 45))
    _require_shape("next_observations", next_observations, (*prefix, 45))
    _require_shape("actions", actions, (*prefix, 12))
    _require_shape("prev_actions", prev_actions, (*prefix, 12))
    _require_shape("commands", commands, (*prefix, 3))
    _require_shape("contacts", contacts, (*prefix, 4))
    _require_shape("dones", tensors["dones"], prefix)
    _require_shape("timeouts", tensors["timeouts"], prefix)
    _require_shape("collector_types", tensors["collector_types"], prefix)
    _require_shape("trace_valid_masks", tensors["trace_valid_masks"], prefix)
    _require_shape("episode_ids", episode_ids, prefix)
    _require_shape("timesteps", timesteps, prefix)

    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Dataset field {name!r} contains NaN or Inf.")

    checks = {
        "obs.angular_velocity": (observations[..., 0:3], states[..., 3:6]),
        "obs.gravity": (observations[..., 3:6], states[..., 6:9]),
        "obs.command": (observations[..., 6:9], commands),
        "obs.previous_action": (observations[..., -12:], prev_actions),
        "next_obs.angular_velocity": (next_observations[..., 0:3], next_states[..., 3:6]),
        "next_obs.gravity": (next_observations[..., 3:6], next_states[..., 6:9]),
    }
    for name, (actual, expected) in checks.items():
        error = float((actual - expected).abs().max())
        if error > 1.0e-5:
            raise ValueError(f"{name} is inconsistent with the state layout; max error={error}.")

    # Exact stateful reward reconstruction requires each stored column to start
    # at a real episode boundary.
    if not torch.equal(timesteps[0], torch.zeros_like(timesteps[0])):
        raise ValueError("Every dataset trajectory must begin at timestep zero for exact reward reconstruction.")

    for env_id in range(num_envs):
        for t in range(1, time_steps):
            same_episode = episode_ids[t, env_id] == episode_ids[t - 1, env_id]
            if same_episode:
                if timesteps[t, env_id] != timesteps[t - 1, env_id] + 1:
                    raise ValueError(f"Non-contiguous timestep at t={t}, env={env_id}.")
            elif timesteps[t, env_id] != 0:
                raise ValueError(f"New episode does not restart at timestep zero at t={t}, env={env_id}.")
    return time_steps, num_envs


def _recompute_one_step_rewards(
    tensors: dict[str, torch.Tensor],
    cfg: FlashSACWorldModelEnvConfig,
    *,
    eligible: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    states = tensors["states"].to(device=device, dtype=torch.float32)
    actions = tensors["actions"].to(device=device, dtype=torch.float32)
    next_states = tensors["next_states"].to(device=device, dtype=torch.float32)
    contacts = tensors["contacts"].to(device=device, dtype=torch.float32)
    commands = tensors["commands"].to(device=device, dtype=torch.float32)
    observations = tensors["observations"].to(device=device, dtype=torch.float32)
    episode_ids = tensors["episode_ids"].to(device=device, dtype=torch.long)
    timesteps = tensors["timesteps"].to(device=device, dtype=torch.long)
    dones = tensors["dones"].to(device=device, dtype=torch.bool)
    timeouts = tensors["timeouts"].to(device=device, dtype=torch.bool)
    eligible = eligible.to(device=device, dtype=torch.bool)

    time_steps, num_envs = states.shape[:2]
    reward_state = _configure_reward_state(
        cfg,
        num_envs=num_envs,
        action_dim=actions.shape[-1],
        device=device,
    )
    rewards = torch.empty(time_steps, num_envs, dtype=torch.float32, device=device)
    term_sums: dict[str, float] = {}
    previous_episode_ids: torch.Tensor | None = None

    for t in range(time_steps):
        episode_start = (
            timesteps[t] == 0
            if previous_episode_ids is None
            else (timesteps[t] == 0) | (episode_ids[t] != previous_episode_ids)
        )
        if episode_start.any():
            reward_state.last_joint_vel[episode_start] = states[t, episode_start, 21:33]
            reward_state.last_action[episode_start] = observations[t, episode_start, -12:]
            reward_state.last_foot_pos_b[episode_start] = 0
            reward_state.last_foot_contact[episode_start] = False
            reward_state.contact_transition_ema[episode_start] = 0
            reward_state.diagonal_foot_velocity_abs_ema[episode_start] = 0
            reward_state.diagonal_foot_velocity_sq_ema[episode_start] = 0
            reward_state.base_lin_vel_xy_ema[episode_start] = states[t, episode_start, 0:2]
            reward_state.base_yaw_vel_ema[episode_start] = states[t, episode_start, 5]

        reward_t, terms = compute_go2_imagination_reward(
            state=next_states[t],
            action=actions[t],
            command=commands[t],
            foot_contact=contacts[t],
            episode_length=timesteps[t],
            reward_state=reward_state,
            epistemic_uncertainty=torch.zeros(num_envs, device=device),
        )
        terminated_t = dones[t] & ~timeouts[t]
        reward_t = reward_t + terminated_t.float() * float(cfg.reward_termination_penalty)
        rewards[t] = torch.where(eligible[t], reward_t, torch.zeros_like(reward_t))
        for name, value in terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + float(
                value.detach()[eligible[t]].sum().cpu()
            )
        previous_episode_ids = episode_ids[t].clone()

    denominator = float(eligible.sum())
    if denominator <= 0:
        raise ValueError("Dataset has no eligible real transitions.")
    return rewards.cpu(), {name: total / denominator for name, total in term_sums.items()}


def _build_n_step(
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_observations: torch.Tensor,
    episode_ids: torch.Tensor,
    timesteps: torch.Tensor,
    eligible: torch.Tensor | None,
    gamma: float,
    n_step: int,
) -> dict[str, torch.Tensor]:
    """Create n-step rows without crossing episode or dataset boundaries."""

    if n_step < 1:
        raise ValueError("n_step must be positive.")
    time_steps, num_envs = rewards.shape
    if eligible is None:
        eligible = torch.ones_like(terminated, dtype=torch.bool)
    if eligible.shape != terminated.shape:
        raise ValueError("eligible mask must have shape [time, env].")
    rows: dict[str, list[torch.Tensor]] = {
        **{key: [] for key in REPLAY_KEYS},
        "source_episode_id": [],
        "source_env_index": [],
        "source_time_index": [],
        "source_timestep": [],
        "n_step_length": [],
        "one_step_reward": [],
    }

    for env_id in range(num_envs):
        for start in range(time_steps):
            if not bool(eligible[start, env_id]):
                continue
            total_reward = torch.zeros((), dtype=torch.float32)
            discount = 1.0
            end = start
            valid = False
            for offset in range(n_step):
                t = start + offset
                if t >= time_steps:
                    break
                if not bool(eligible[t, env_id]):
                    break
                if offset > 0:
                    same_episode = episode_ids[t, env_id] == episode_ids[t - 1, env_id]
                    contiguous = timesteps[t, env_id] == timesteps[t - 1, env_id] + 1
                    if not bool(same_episode and contiguous):
                        break
                total_reward = total_reward + discount * rewards[t, env_id]
                end = t
                done = bool(terminated[t, env_id] or truncated[t, env_id])
                if done or offset == n_step - 1:
                    valid = True
                    break
                discount *= gamma
            if not valid:
                continue

            rows["observation"].append(observations[start, env_id])
            rows["action"].append(actions[start, env_id])
            rows["reward"].append(total_reward)
            rows["terminated"].append(terminated[end, env_id].float())
            rows["truncated"].append(truncated[end, env_id].float())
            rows["next_observation"].append(next_observations[end, env_id])
            rows["source_episode_id"].append(episode_ids[start, env_id])
            rows["source_env_index"].append(torch.tensor(env_id, dtype=torch.long))
            rows["source_time_index"].append(torch.tensor(start, dtype=torch.long))
            rows["source_timestep"].append(timesteps[start, env_id])
            rows["n_step_length"].append(torch.tensor(end - start + 1, dtype=torch.long))
            rows["one_step_reward"].append(rewards[start, env_id])

    if not rows["reward"]:
        raise ValueError("No valid n-step transitions were produced.")
    return {key: torch.stack(values).contiguous() for key, values in rows.items()}


def build_real_replay(
    *,
    dataset: dict[str, Any],
    config: Any,
    condition: str,
    dataset_size: str,
    source_dataset_path: Path,
    source_dataset_sha256: str,
    config_path: Path,
    config_sha256: str,
    device: torch.device,
    allowed_collector_types: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    required = (
        "states",
        "actions",
        "next_states",
        "contacts",
        "observations",
        "next_observations",
        "commands",
        "dones",
        "timeouts",
        "prev_actions",
        "episode_ids",
        "timesteps",
        "collector_types",
        "trace_valid_masks",
    )
    tensors = {key: _stack_time_key(dataset, key) for key in required}
    time_steps, num_envs = _validate_dataset_layout(tensors)

    wm_dict = OmegaConf.to_container(config.world_model, resolve=True, throw_on_missing=True)
    if not isinstance(wm_dict, dict):
        raise TypeError("config.world_model must resolve to a mapping.")
    wm_cfg = FlashSACWorldModelEnvConfig(**wm_dict)
    gamma = float(config.agent.gamma)
    n_step = int(config.agent.n_step)

    states = tensors["states"].float()
    next_states = tensors["next_states"].float()
    policy_observations = tensors["observations"].float()
    policy_next_observations = tensors["next_observations"].float()
    critic_observations = torch.cat([states[..., 0:3], policy_observations], dim=-1)
    critic_next_observations = torch.cat(
        [next_states[..., 0:3], policy_next_observations],
        dim=-1,
    )

    timeouts = tensors["timeouts"].bool()
    dones = tensors["dones"].bool()
    terminated = dones & ~timeouts
    allowed_collectors = tuple(int(value) for value in allowed_collector_types)
    if not allowed_collectors:
        raise ValueError("allowed_collector_types must not be empty.")
    collector_types = tensors["collector_types"].long()
    collector_allowed = torch.zeros_like(collector_types, dtype=torch.bool)
    for collector_type in allowed_collectors:
        collector_allowed |= collector_types == collector_type
    if not bool(collector_allowed.all()):
        actual = sorted(int(value) for value in torch.unique(collector_types).tolist())
        raise ValueError(
            "Real replay contains collector types outside the explicitly allowed set: "
            f"actual={actual}, allowed={sorted(allowed_collectors)}."
        )
    if not bool(tensors["trace_valid_masks"].bool().all()):
        raise ValueError("Formal V12 real replay requires every selected transition to be trace-valid.")

    # The ordinary expert collector uses auto-reset.  On a done row its
    # next_state/next_observation are already the reset state, not the final
    # transition state.  Such a row cannot be used to recompute the public
    # reward and is excluded, together with any n-step window that needs it.
    eligible = ~dones
    one_step_rewards, reward_term_means = _recompute_one_step_rewards(
        tensors,
        wm_cfg,
        eligible=eligible,
        device=device,
    )
    policy_mask = tuple(int(index) for index in wm_cfg.policy_action_mask_indices)
    keep = [index for index in range(12) if index not in set(policy_mask)]
    if not keep:
        raise ValueError("policy_action_mask_indices removes every action.")
    policy_actions = tensors["actions"].float()[..., keep]

    replay = _build_n_step(
        observations=critic_observations,
        actions=policy_actions,
        rewards=one_step_rewards,
        terminated=terminated,
        truncated=timeouts,
        next_observations=critic_next_observations,
        episode_ids=tensors["episode_ids"].long(),
        timesteps=tensors["timesteps"].long(),
        eligible=eligible,
        gamma=gamma,
        n_step=n_step,
    )

    reward_config = {
        key: value
        for key, value in asdict(wm_cfg).items()
        if key.startswith("reward_") or key in {"step_dt", "uncertainty_penalty_weight"}
    }
    metadata = {
        "condition": str(condition),
        "dataset_size": str(dataset_size),
        "source_dataset_path": str(source_dataset_path),
        "source_dataset_sha256": source_dataset_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "reward_config": reward_config,
        "reward_config_sha256": _canonical_sha256(reward_config),
        "reward_version": wm_cfg.reward_version,
        "critic_observation_dim": 48,
        "actor_observation_dim": 45,
        "full_action_dim": 12,
        "action_dim": len(keep),
        "policy_action_mask_indices": list(policy_mask),
        "gamma": gamma,
        "n_step": n_step,
        "source_time_steps": time_steps,
        "source_num_envs": num_envs,
        "source_num_transitions": time_steps * num_envs,
        "source_collector_types": sorted(
            int(value) for value in torch.unique(collector_types).tolist()
        ),
        "allowed_collector_types": sorted(allowed_collectors),
        "source_eligible_transitions": int(eligible.sum()),
        "excluded_autoreset_done_transitions": int(dones.sum()),
        "num_transitions": int(replay["reward"].shape[0]),
        "one_step_reward_mean": float(one_step_rewards[eligible].mean()),
        "one_step_reward_std": float(one_step_rewards[eligible].std(unbiased=False)),
        "n_step_reward_mean": float(replay["reward"].mean()),
        "n_step_reward_std": float(replay["reward"].std(unbiased=False)),
        "source_terminated_count": int(terminated.sum()),
        "source_truncated_count": int(timeouts.sum()),
        "replay_terminated_count": int(replay["terminated"].sum()),
        "replay_truncated_count": int(replay["truncated"].sum()),
        "dynamics_termination_target_count": int(
            _stack_time_key(dataset, "terminations").float().gt(0.5).sum()
        )
        if "terminations" in dataset
        else None,
        "reward_term_means": reward_term_means,
        "contact_alignment": "post_step_contact_aligned_with_next_state",
        "source_reward_used": False,
        "epistemic_uncertainty_for_real": 0.0,
    }
    artifact = {
        "format_version": FORMAT_VERSION,
        **replay,
        "metadata": metadata,
    }
    for key in REPLAY_KEYS:
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.shape[0] != metadata["num_transitions"]:
            raise AssertionError(f"Invalid replay tensor {key!r}.")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError(f"Replay tensor {key!r} contains NaN or Inf.")
    if artifact["observation"].shape[-1] != 48 or artifact["next_observation"].shape[-1] != 48:
        raise AssertionError("Critic replay observations are not 48-dimensional.")
    return artifact


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    device = torch.device(args.device)

    dataset_sha256 = _sha256_file(dataset_path)
    config_sha256 = _sha256_file(config_path)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    artifact = build_real_replay(
        dataset=dataset,
        config=config,
        condition=args.condition,
        dataset_size=args.dataset_size,
        source_dataset_path=dataset_path,
        source_dataset_sha256=dataset_sha256,
        config_path=config_path,
        config_sha256=config_sha256,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(artifact, temporary_path)
    temporary_path.replace(output_path)
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))
    print(f"[V12 real replay] wrote {output_path}")
    print(f"[V12 real replay] sha256={_sha256_file(output_path)}")


if __name__ == "__main__":
    main()
