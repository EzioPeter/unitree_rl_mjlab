"""Audit a V12 real replay artifact against its source dataset and reward config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.reinforcement_learning.rwm_flashsac.build_go2_real_replay import (
    FORMAT_VERSION,
    REPLAY_KEYS,
    _configure_reward_state,
    _recompute_one_step_rewards,
    _sha256_file,
    _stack_time_key,
    _validate_dataset_layout,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env import (
    FlashSACWorldModelEnvConfig,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _args()
    artifact_path = Path(args.artifact_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)

    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unexpected format: {artifact.get('format_version')!r}")
    metadata = artifact.get("metadata") or {}
    if metadata.get("source_dataset_sha256") != _sha256_file(dataset_path):
        raise ValueError("Source dataset SHA256 mismatch.")
    if metadata.get("config_sha256") != _sha256_file(config_path):
        raise ValueError("Config SHA256 mismatch.")

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
    _validate_dataset_layout(tensors)
    dones = tensors["dones"].bool()
    timeouts = tensors["timeouts"].bool()
    eligible = ~dones

    wm_dict = OmegaConf.to_container(config.world_model, resolve=True, throw_on_missing=True)
    assert isinstance(wm_dict, dict)
    wm_cfg = FlashSACWorldModelEnvConfig(**wm_dict)
    # Construct once as an explicit mapping smoke test.  Reward recomputation
    # below constructs another state and exercises the full sequence.
    configured_state = _configure_reward_state(
        wm_cfg,
        num_envs=int(tensors["states"].shape[1]),
        action_dim=12,
        device=torch.device(args.device),
    )
    if configured_state.reward_version != wm_cfg.reward_version:
        raise AssertionError("Reward version was not propagated.")

    one_step_rewards, _ = _recompute_one_step_rewards(
        tensors,
        wm_cfg,
        eligible=eligible,
        device=torch.device(args.device),
    )

    count = int(artifact["reward"].shape[0])
    for key in REPLAY_KEYS:
        value = artifact.get(key)
        if not isinstance(value, torch.Tensor) or int(value.shape[0]) != count:
            raise ValueError(f"Replay field {key!r} has an invalid leading dimension.")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Replay field {key!r} contains NaN/Inf.")

    env_index = artifact["source_env_index"].long()
    start = artifact["source_time_index"].long()
    source_timestep = artifact["source_timestep"].long()
    source_episode = artifact["source_episode_id"].long()
    lengths = artifact["n_step_length"].long()
    end = start + lengths - 1
    if not bool((lengths == int(config.agent.n_step)).all()):
        raise ValueError("Formal artifact contains shortened non-terminal n-step rows.")
    if not bool(eligible[start, env_index].all()):
        raise ValueError("Artifact contains an ineligible auto-reset transition.")
    if not torch.equal(tensors["episode_ids"][start, env_index].long(), source_episode):
        raise ValueError("source_episode_id does not map back to the dataset.")
    if not torch.equal(tensors["timesteps"][start, env_index].long(), source_timestep):
        raise ValueError("source_timestep does not map back to the dataset.")

    expected_observation = torch.cat(
        [
            tensors["states"][start, env_index, 0:3].float(),
            tensors["observations"][start, env_index].float(),
        ],
        dim=-1,
    )
    expected_next_observation = torch.cat(
        [
            tensors["next_states"][end, env_index, 0:3].float(),
            tensors["next_observations"][end, env_index].float(),
        ],
        dim=-1,
    )
    policy_mask = set(int(index) for index in wm_cfg.policy_action_mask_indices)
    keep = [index for index in range(12) if index not in policy_mask]
    expected_action = tensors["actions"][start, env_index].float()[:, keep]

    gamma = float(config.agent.gamma)
    expected_reward = torch.zeros(count)
    for offset in range(int(config.agent.n_step)):
        expected_reward += (gamma**offset) * one_step_rewards[start + offset, env_index]

    errors = {
        "observation_max_abs": float((artifact["observation"] - expected_observation).abs().max()),
        "next_observation_max_abs": float(
            (artifact["next_observation"] - expected_next_observation).abs().max()
        ),
        "action_max_abs": float((artifact["action"] - expected_action).abs().max()),
        "n_step_reward_max_abs": float((artifact["reward"] - expected_reward).abs().max()),
        "one_step_reward_max_abs": float(
            (artifact["one_step_reward"] - one_step_rewards[start, env_index]).abs().max()
        ),
    }
    if any(value > 2.0e-6 for value in errors.values()):
        raise ValueError(f"Artifact/source mismatch: {errors}")

    expected_terminated = (dones & ~timeouts)[end, env_index].float()
    expected_truncated = timeouts[end, env_index].float()
    if not torch.equal(artifact["terminated"], expected_terminated):
        raise ValueError("terminated semantics mismatch.")
    if not torch.equal(artifact["truncated"], expected_truncated):
        raise ValueError("truncated semantics mismatch.")

    source_rewards = _stack_time_key(dataset, "rewards").float()
    source_reward_at_rows = source_rewards[start, env_index]
    source_vs_recomputed_abs = (
        source_reward_at_rows - artifact["one_step_reward"]
    ).abs()
    source_reward_was_not_copied = not torch.allclose(
        source_reward_at_rows,
        artifact["one_step_reward"],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    if not source_reward_was_not_copied:
        raise ValueError("Artifact one-step reward is an unchanged copy of the source task reward.")
    critic_observation_dim = int(
        metadata.get("critic_observation_dim", artifact["observation"].shape[-1])
    )
    actor_observation_dim = int(
        metadata.get("actor_observation_dim", critic_observation_dim - 3)
    )
    if critic_observation_dim != int(artifact["observation"].shape[-1]):
        raise ValueError("Replay metadata critic observation dimension is inconsistent.")
    if actor_observation_dim not in {
        critic_observation_dim,
        critic_observation_dim - 3,
    }:
        raise ValueError("Replay metadata actor observation dimension is unsupported.")
    report = {
        "status": "PASS",
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "num_transitions": count,
        "critic_observation_dim": critic_observation_dim,
        "actor_observation_dim": actor_observation_dim,
        "action_dim": int(artifact["action"].shape[-1]),
        "n_step": int(config.agent.n_step),
        "gamma": gamma,
        "n_step_length_values": sorted(set(int(v) for v in lengths.tolist())),
        "source_dataset_reward_mean": float(source_rewards.mean()),
        "recomputed_one_step_reward_mean": float(one_step_rewards[eligible].mean()),
        "source_vs_recomputed_reward_mean_abs": float(source_vs_recomputed_abs.mean()),
        "source_vs_recomputed_reward_max_abs": float(source_vs_recomputed_abs.max()),
        "source_reward_was_not_copied": source_reward_was_not_copied,
        "excluded_autoreset_done_rows": int(dones.sum()),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
